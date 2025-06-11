package main

/*
#cgo windows LDFLAGS: -L${SRCDIR}/lib -lwhisper -lstdc++ -lm
#cgo windows CFLAGS: -I${SRCDIR}/include
#include <stdlib.h>
#include "whisper.h"
*/
import "C"

// #cgo LDFLAGS: -L${SRCDIR}/lib -lwhisper -lm

import (
	"fmt"
	"github.com/gordonklaus/portaudio"
	_ "github.com/mitchellh/go-ps"
	"golang.org/x/term"
	"log"
	"math"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

const (
	sampleRate  = 16000
	bufferSize  = 1024
	numChannels = 1
	//modelPath      = "models/ggml-small.bin" // Путь к модели Whisper
	modelPath      = "models/ggml-base.bin" // Путь к модели Whisper
	vadThreshold   = 0.01                   // Порог VAD (0-1)
	minSpeechMs    = 500                    // Минимальная длительность речи (мс)
	silenceTimeout = 2000                   // Таймаут молчания (мс)
)

var (
	ctx          *C.struct_whisper_context
	audioBuffer  []float32
	bufferMutex  sync.Mutex
	isRecording  bool = true
	lastSpeechTs time.Time
)

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

	//fmt.Println("Recording... Press Ctrl+C to stop.")
	//select {}

	// Канал для обработки сигналов
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Горутина для обработки горячих клавиш
	go handleHotkeys()

	fmt.Println("Recording started. Press:")
	fmt.Println("- Space: Pause/Resume")
	fmt.Println("- Q: Quit")
	fmt.Println("- C: Clear buffer")

	// Ожидаем сигнал завершения
	<-sigChan
	fmt.Println("\nShutting down...")
}

func processAudioOld(in []float32) {
	// Добавляем новые сэмплы в буфер
	audioBuffer = append(audioBuffer, in...)

	// Если в буфере достаточно данных (например, 3 секунды аудио), обрабатываем их
	if len(audioBuffer) >= sampleRate*3 {
		processBuffer()
	}
}

func processAudio(in []float32) {
	bufferMutex.Lock()
	defer bufferMutex.Unlock()

	if !isRecording {
		return
	}

	// VAD - проверка наличия речи
	if hasSpeech(in) {
		lastSpeechTs = time.Now()
		audioBuffer = append(audioBuffer, in...)
		processIncremental() // Потоковая обработка
	} else if time.Since(lastSpeechTs) > time.Millisecond*silenceTimeout && len(audioBuffer) > 0 {
		processBuffer() // Обработка при длительном молчании
	}
}

// VAD - простая реализация детектора речи
func hasSpeech(samples []float32) bool {
	var energy float32
	for _, s := range samples {
		energy += s * s
	}
	rms := float32(0.0)
	if len(samples) > 0 {
		rms = float32(math.Sqrt(float64(energy / float32(len(samples)))))
	}
	return rms > vadThreshold
}

// Потоковая обработка с частичными результатами
func processIncremental() {
	if len(audioBuffer) < sampleRate { // Минимум 1 секунда для обработки
		return
	}

	params := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)
	params.language = C.CString("ru") // Язык русский
	defer C.free(unsafe.Pointer(params.language))
	params.translate = false
	params.no_context = true
	params.single_segment = false
	params.print_realtime = true
	params.print_progress = false

	result := C.whisper_full(ctx, params, (*C.float)(&audioBuffer[0]), C.int(len(audioBuffer)))
	if result != 0 {
		log.Printf("Ошибка обработки: %d", result)
		return
	}

	// Вывод только новых сегментов
	n_segments := C.whisper_full_n_segments(ctx)
	for i := int(n_segments) - 1; i >= 0; i-- {
		text := C.whisper_full_get_segment_text(ctx, C.int(i))
		fmt.Printf("\r%s", C.GoString(text)) // \r для перезаписи строки
	}
}

func processBufferOld() {
	// Копируем буфер для обработки
	buffer := make([]float32, len(audioBuffer))
	copy(buffer, audioBuffer)

	// Очищаем буфер для новых данных
	audioBuffer = nil

	// Создаем параметры для Whisper
	params := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)

	// Set language (e.g., "en" for English)
	params.language = C.CString("ru") // Change to your desired language code
	defer C.free(unsafe.Pointer(params.language))

	// Force transcription in the specified language (disable auto-detection)
	params.detect_language = false //Если хотите жестко задать язык
	params.translate = false       // Отключаем перевод (если включен)
	params.no_context = true
	params.single_segment = false

	// Выполняем транскрибацию
	start := time.Now()
	result := C.whisper_full(ctx, params, (*C.float)(&buffer[0]), C.int(len(buffer)))
	if result != 0 {
		log.Printf("Whisper processing failed with code %d", result)
		return
	}

	// Получаем результаты
	n_segments := C.whisper_full_n_segments(ctx)
	for i := 0; i < int(n_segments); i++ {
		text := C.whisper_full_get_segment_text(ctx, C.int(i))
		t0 := C.whisper_full_get_segment_t0(ctx, C.int(i))
		t1 := C.whisper_full_get_segment_t1(ctx, C.int(i))

		fmt.Printf("[%.2fs->%.2fs] %s\n",
			float64(t0)/100.0,
			float64(t1)/100.0,
			C.GoString(text))
	}

	fmt.Printf("Processing took: %v\n", time.Since(start))
}

// Полная обработка буфера
func processBuffer() {
	if len(audioBuffer) == 0 {
		return
	}

	params := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)
	params.language = C.CString("ru")
	defer C.free(unsafe.Pointer(params.language))
	params.translate = false

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

// Обработка горячих клавиш
func handleHotkeys() {
	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		return
	}
	defer term.Restore(int(os.Stdin.Fd()), oldState)

	b := make([]byte, 1)
	for {
		os.Stdin.Read(b)
		fmt.Println(b[0])
		switch b[0] {
		case ' ': // Пауза/продолжение
			bufferMutex.Lock()
			isRecording = !isRecording
			status := "resumed"
			if !isRecording {
				status = "paused"
			}
			fmt.Printf("\nRecording %s\n", status)
			bufferMutex.Unlock()
		case 'q', 'Q': // Выход
			pid := os.Getpid()
			process, _ := os.FindProcess(pid)
			process.Signal(syscall.SIGINT)
			return
		case 'c', 'C': // Очистка буфера
			bufferMutex.Lock()
			audioBuffer = nil
			fmt.Println("\nBuffer cleared")
			bufferMutex.Unlock()
		}
	}
}
