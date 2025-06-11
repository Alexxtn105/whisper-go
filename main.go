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
	"log"
	"sync"
	"time"
	"unsafe"
)

const (
	sampleRate  = 16000
	bufferSize  = 1024
	numChannels = 1
	//modelPath      = "models/ggml-small.bin" // Путь к модели Whisper
	modelPath      = "models/ggml-base.bin" // Путь к модели Whisper
	vadThreshold   = 0.5                    // Порог VAD (0-1)
	minSpeechMs    = 300                    // Минимальная длительность речи (мс)
	silenceTimeout = 1500                   // Таймаут молчания (мс)
)

var (
	ctx          *C.struct_whisper_context
	audioBuffer  []float32
	bufferMutex  sync.Mutex
	isRecording  bool = true
	lastSpeechTs time.Time
)

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

	fmt.Println("Recording... Press Ctrl+C to stop.")
	select {}
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

}

func processAudio(in []float32) {
	// Добавляем новые сэмплы в буфер
	audioBuffer = append(audioBuffer, in...)

	// Если в буфере достаточно данных (например, 3 секунды аудио), обрабатываем их
	if len(audioBuffer) >= sampleRate*3 {
		processBuffer()
	}
}

func processBuffer() {
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
