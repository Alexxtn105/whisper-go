package main

/*
#cgo windows LDFLAGS: -L${SRCDIR}/lib -lwhisper -lstdc++ -lm
#cgo windows CFLAGS: -I${SRCDIR}/include
#include <stdlib.h>
#include "whisper.h"
*/
import "C"
import (
	"fmt"
	"github.com/eiannone/keyboard"
	"github.com/gordonklaus/portaudio"

	"log"
	"math"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

const (
	sampleRate        = 16000
	bufferSize        = 1024
	numChannels       = 1
	modelPath         = "models/ggml-base.bin" // Путь к модели Whisper
	vadThreshold      = 0.01                   // Порог VAD (0-1)
	silenceTimeout    = 2000                   // Таймаут молчания (мс)
	processingTimeout = 10 * time.Second
	maxBufferSize     = sampleRate * 30 // Максимальный размер буфера 30 секунд
)

var (
	ctx          *C.struct_whisper_context
	audioBuffer  []float32
	bufferMutex  sync.Mutex
	isRecording  atomic.Bool
	lastSpeechTs time.Time
	speechActive bool
	processing   atomic.Bool
)

func init() {
	isRecording.Store(true)
}

func initWhisper() {
	modelPathC := C.CString(modelPath)
	defer C.free(unsafe.Pointer(modelPathC))

	params := C.struct_whisper_context_params{
		use_gpu: true,
	}

	ctx = C.whisper_init_from_file_with_params(modelPathC, params)
	if ctx == nil {
		log.Fatal("Failed to initialize whisper context")
	}
	fmt.Println("Whisper model loaded successfully")
}

func main() {
	//region Инициализируем Whisper
	initWhisper()
	defer C.whisper_free(ctx)
	//endregion

	//region Инициализируем PortAudio
	err := portaudio.Initialize()
	if err != nil {
		log.Fatalf("Failed to initialize PortAudio: %v", err)
	}
	defer func() {
		err := portaudio.Terminate()
		if err != nil {
			log.Fatalf("Failed to terminate PortAudio: %v", err)
		}
	}()
	// Создаем поток для захвата аудио
	stream, err := portaudio.OpenDefaultStream(numChannels, 0, sampleRate, bufferSize, processAudio)
	if err != nil {
		log.Fatalf("Failed to open audio stream: %v", err)
	}
	defer func(stream *portaudio.Stream) {
		err := stream.Close()
		if err != nil {
			log.Fatalf("Failed to close audio stream: %v", err)
		}
	}(stream)
	//endregion

	//region Начинаем захват аудио
	err = stream.Start()
	if err != nil {
		log.Fatalf("Failed to start audio stream: %v", err)
	}
	defer func(stream *portaudio.Stream) {
		err := stream.Stop()
		if err != nil {
			log.Fatalf("Failed to stop audio stream: %v", err)
		}
	}(stream)
	//endregion

	//region Инициализация клавиатуры
	if err := keyboard.Open(); err != nil {
		log.Fatal(err)
	}
	defer func() {
		err := keyboard.Close()
		if err != nil {
			log.Fatalf("Failed close keyboard: %v", err)
		}
	}()
	//endregion

	fmt.Println("Recording started. Controls:")
	fmt.Println("- Space: Pause/Resume recording")
	fmt.Println("- C: Clear buffer")
	fmt.Println("- Q: Quit")

	// Канал для сигналов завершения
	done := make(chan struct{})
	defer close(done)

	// Горутина для обработки клавиш
	go func() {
		for {
			select {
			case <-done:
				return
			default:
				char, key, err := keyboard.GetKey()
				if err != nil {
					continue
				}

				switch {
				case key == keyboard.KeySpace:
					bufferMutex.Lock()
					old := isRecording.Swap(!isRecording.Load())
					_ = old
					//isRecording = !isRecording
					status := "RESUMED"
					if !isRecording.Load() {
						status = "PAUSED"
					}
					fmt.Printf("\nRecording %s\n", status)
					bufferMutex.Unlock()

				case char == 'c' || char == 'C' || char == 'с' || char == 'С':
					bufferMutex.Lock()
					audioBuffer = nil
					fmt.Println("\nBuffer cleared")
					bufferMutex.Unlock()

				case char == 'q' || char == 'Q' || char == 'й' || char == 'Й':
					fmt.Println("\nQuitting...")
					os.Exit(0)
				}
			}
		}
	}()

	// Ожидание Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan
	fmt.Println("\nShutting down...")
}

func processAudio(in []float32) {
	if !isRecording.Load() {
		return
	}

	// Блокировка для записи
	bufferMutex.Lock()
	defer bufferMutex.Unlock()

	// Добавляем новые сэмплы с проверкой переполнения
	audioBuffer = append(audioBuffer, in...)
	if len(audioBuffer) > maxBufferSize {
		audioBuffer = audioBuffer[len(audioBuffer)-maxBufferSize:]
	}

	// Анализ только последних 300 мс для VAD
	vadWindow := 300 * sampleRate / 1000
	if len(audioBuffer) < vadWindow {
		return
	}

	window := audioBuffer[len(audioBuffer)-vadWindow:]
	rms := computeRMS(window)

	// Обновление статуса речи
	if rms > vadThreshold {
		lastSpeechTs = time.Now()
		if !speechActive {
			speechActive = true
			fmt.Println("\nSpeech detected, recording...")
		}
		return
	}

	// Обработка окончания речи
	if speechActive && time.Since(lastSpeechTs) > silenceTimeout*time.Millisecond {
		speechActive = false
		fmt.Println("\nSilence detected, starting processing...")
		go processBuffer() // Асинхронная обработка
	}
}

func computeRMS(samples []float32) float64 {
	var sum float64
	for _, s := range samples {
		sum += float64(s * s)
	}
	return math.Sqrt(sum / float64(len(samples)))
}

// processBuffer Полная обработка буфера
func processBuffer() {
	// Проверяем, не обрабатывается ли уже другой фрагмент
	if !processing.CompareAndSwap(false, true) {
		log.Println("Previous processing still in progress, skipping")
		return
	}
	defer processing.Store(false)

	// Быстрое копирование буфера с минимальной блокировкой
	bufferMutex.Lock()
	if len(audioBuffer) == 0 {
		bufferMutex.Unlock()
		return
	}

	processingBuffer := make([]float32, len(audioBuffer))
	copy(processingBuffer, audioBuffer)
	audioBuffer = nil
	bufferMutex.Unlock()

	// Канал для результатов обработки
	resultChan := make(chan struct{}, 1)
	defer close(resultChan)

	// Обработка в отдельной горутине
	go func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Whisper processing panic: %v", r)
			}
			resultChan <- struct{}{}
		}()

		params := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)
		params.language = C.CString("ru")
		defer C.free(unsafe.Pointer(params.language))
		params.translate = false
		params.no_context = true
		params.single_segment = true

		log.Printf("Starting processing of %.2f seconds audio", float64(len(processingBuffer))/sampleRate)
		start := time.Now()

		// Важная проверка перед вызовом C-кода
		if len(processingBuffer) == 0 {
			log.Println("Empty buffer passed to Whisper")
			return
		}

		ret := C.whisper_full(
			ctx,
			params,
			(*C.float)(&processingBuffer[0]),
			C.int(len(processingBuffer)),
		)
		if ret != 0 {
			log.Printf("Whisper processing failed with code %d", ret)
			return
		}

		nSegments := C.whisper_full_n_segments(ctx)
		for i := 0; i < int(nSegments); i++ {
			text := C.whisper_full_get_segment_text(ctx, C.int(i))
			fmt.Printf("\n[%.2fs] %s\n", float64(C.whisper_full_get_segment_t1(ctx, C.int(i)))/100.0,
				C.GoString(text))
		}
		log.Printf("Processing completed in %v", time.Since(start))
	}()

	// Таймаут для обработки
	select {
	case <-resultChan:
		log.Println("Processing finished successfully")
	case <-time.After(processingTimeout):
		log.Println("Processing timeout reached, terminating")
		// Важно: НЕ прерываем C-код, просто продолжаем работу
	}
}
