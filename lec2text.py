import os
import sys
import shutil
from pathlib import Path

import torch
import whisper


# =========================
# 설정
# =========================
MODEL_NAME = "medium"
LANGUAGE = "ko"
SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


# =========================
# 유틸 함수
# =========================
def check_ffmpeg():
    """FFmpeg 설치 여부 확인"""
    if shutil.which("ffmpeg") is None:
        print("❌ FFmpeg를 찾을 수 없습니다.")
        print("   FFmpeg를 설치하고 PATH 환경변수에 등록해주세요.")
        sys.exit(1)


def get_video_files():
    """현재 디렉터리에서 지원되는 영상 파일 목록 가져오기"""
    files = sorted(
        [f for f in os.listdir() if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS]
    )
    return files


def choose_video_file(files):
    """사용자로부터 영상 파일 선택받기"""
    if not files:
        print("❌ 현재 디렉터리에 변환할 영상 파일이 없습니다.")
        sys.exit(1)

    print("변환할 영상 파일을 선택하세요:")
    for i, file in enumerate(files, start=1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = int(input("번호 입력: ").strip())
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print("❌ 목록에 있는 번호를 입력해주세요.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")


def choose_progress_mode():
    """
    진행 표시 방식 선택
    1. progress bar 중심
    2. segment 텍스트 출력 중심
    """
    print("\n진행 표시 방식을 선택하세요:")
    print("1. 진행률 바 표시 (권장)")
    print("2. 인식된 세그먼트 텍스트를 콘솔에 출력")

    while True:
        mode = input("번호 입력: ").strip()
        if mode == "1":
            return "bar"
        elif mode == "2":
            return "segments"
        else:
            print("❌ 1 또는 2를 입력해주세요.")


def select_device():
    """CUDA 가능 여부에 따라 device / fp16 설정"""
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False


def format_timestamp(seconds):
    """초 단위를 HH:MM:SS 형식 문자열로 변환"""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def format_srt_timestamp(seconds):
    """초 단위를 SRT 형식(HH:MM:SS,mmm) 문자열로 변환"""
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def save_plain_text(output_path, text):
    """순수 텍스트 저장"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text.strip())


def save_segment_text(output_path, segments):
    """타임스탬프 포함 텍스트 저장"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()

            f.write(f"[{i}] {start} ~ {end}\n")
            f.write(f"{text}\n\n")


def save_srt(output_path, segments):
    """SRT 자막 파일 저장"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_srt_timestamp(segment["start"])
            end = format_srt_timestamp(segment["end"])
            text = segment["text"].strip()

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


# =========================
# 메인 로직
# =========================
def main():
    check_ffmpeg()

    video_files = get_video_files()
    input_video = choose_video_file(video_files)

    base_name = Path(input_video).stem
    save_dir = Path("result") / base_name
    save_dir.mkdir(parents=True, exist_ok=True)

    output_txt = save_dir / f"{base_name}.txt"
    output_segment_txt = save_dir / f"{base_name}_segments.txt"
    output_srt = save_dir / f"{base_name}.srt"

    progress_mode = choose_progress_mode()

    device, fp16 = select_device()
    print(f"\n🧠 Whisper 모델 로딩 중...")
    print(f"   모델: {MODEL_NAME}")
    print(f"   언어: {LANGUAGE}")
    print(f"   장치: {device.upper()}")

    try:
        model = whisper.load_model(MODEL_NAME, device=device)
    except Exception as e:
        print(f"❌ Whisper 모델 로드 중 오류가 발생했습니다: {e}")
        sys.exit(1)

    print(f"\n🎧 '{input_video}' 음성 텍스트화 진행 중...")

    try:
        if progress_mode == "bar":
            # 일반적으로 progress bar가 표시되는 방식
            result = model.transcribe(
                input_video,
                language=LANGUAGE,
                fp16=fp16,
                verbose=False
            )
        else:
            # 세그먼트 텍스트를 콘솔에 출력하는 방식
            result = model.transcribe(
                input_video,
                language=LANGUAGE,
                fp16=fp16,
                verbose=True
            )
    except Exception as e:
        print(f"❌ 변환 중 오류가 발생했습니다: {e}")
        sys.exit(1)

    full_text = result.get("text", "").strip()
    segments = result.get("segments", [])

    try:
        save_plain_text(output_txt, full_text)
        save_segment_text(output_segment_txt, segments)
        save_srt(output_srt, segments)
    except Exception as e:
        print(f"❌ 결과 저장 중 오류가 발생했습니다: {e}")
        sys.exit(1)

    print("\n✅ 변환 완료!")
    print(f"📄 일반 텍스트: {output_txt}")
    print(f"🕒 타임스탬프 텍스트: {output_segment_txt}")
    print(f"🎬 SRT 자막: {output_srt}")


if __name__ == "__main__":
    main()