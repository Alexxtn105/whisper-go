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
	minSpeechMs       = 500                    // Минимальная длительность речи (мс)
	silenceTimeout    = 2000                   // Таймаут молчания (мс)
	processingTimeout = 10 * time.Second
	maxBufferSize     = sampleRate * 30 // Максимальный размер буфера 30 секунд
)

var (
	ctx         *C.struct_whisper_context
	audioBuffer []float32
	bufferMutex sync.Mutex
	//isRecording   atomic.Bool
	keyboardMutex sync.Mutex
	isRecording   = true
	lastSpeechTs  time.Time
	speechActive  bool
	processing    atomic.Bool
)

func init() {
	//	isRecording.Store(true)
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
	// Инициализируем Whisper
	initWhisper()
	defer C.whisper_free(ctx)

	// Инициализируем PortAudio
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

	// Начинаем захват аудио
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

	// Инициализация клавиатуры
	if err := keyboard.Open(); err != nil {
		log.Fatal(err)
	}
	defer func() {
		err := keyboard.Close()
		if err != nil {
			log.Fatalf("Failed close keyboard: %v", err)
		}
	}()

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
					isRecording = !isRecording
					status := "RESUMED"
					if !isRecording {
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
	bufferMutex.Lock()
	defer bufferMutex.Unlock()

	if !isRecording {
		return
	}

	// Отладочный вывод уровня звука
	rms := computeRMS(in)
	fmt.Printf("\rRMS: %.5f (Threshold: %.5f)", rms, vadThreshold) // Курсор остаётся на строке

	if rms > vadThreshold {
		if !speechActive {
			fmt.Println("\nSpeech detected!")
			speechActive = true
		}
		lastSpeechTs = time.Now()
		audioBuffer = append(audioBuffer, in...)
		if len(audioBuffer) > maxBufferSize {
			audioBuffer = audioBuffer[len(audioBuffer)-maxBufferSize:]
		}
	} else {
		if speechActive && time.Since(lastSpeechTs) > time.Millisecond*silenceTimeout {
			fmt.Println("\nSilence detected, processing...")
			processBuffer()
			speechActive = false
		}
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
	defer func() {
		if processing.Load() {
			processing.Store(false)
		}
	}()

	if len(audioBuffer) == 0 {
		return
	}

	params := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)
	params.language = C.CString("ru")
	defer C.free(unsafe.Pointer(params.language))
	params.translate = false
	params.print_progress = true
	params.print_realtime = true

	result := C.whisper_full(ctx, params, (*C.float)(&audioBuffer[0]), C.int(len(audioBuffer)))
	if result != 0 {
		log.Printf("Ошибка обработки: %d", result)
		return
	}

	n_segments := C.whisper_full_n_segments(ctx)
	for i := 0; i < int(n_segments); i++ {
		text := C.whisper_full_get_segment_text(ctx, C.int(i))
		fmt.Printf("\n%s\n", C.GoString(text)) // \n для новой строки
	}

	audioBuffer = nil
}
