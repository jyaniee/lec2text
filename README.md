# Whisper 기반 영상 음성 텍스트 변환 도구

개인적으로 사용하려고 만든 간단한 영상 음성 텍스트 변환 스크립트.

영상 파일의 오디오를 Whisper로 인식하여 텍스트로 변환하고, 일반 텍스트 / 타임스탬프 텍스트 / SRT 자막 파일로 저장함.

## 구성 파일

| 파일명           | 설명                                       |
| ------------- | ---------------------------------------- |
| `lec2text.py` | Whisper를 이용해 영상 음성을 텍스트로 변환하는 메인 스크립트    |
| `testcuda.py` | PyTorch에서 CUDA(GPU)를 사용할 수 있는지 확인하는 스크립트 |


## 요구 사항

* Python
* `openai-whisper`
* PyTorch
* FFmpeg
* NVIDIA GPU 권장 (CUDA 사용 시 처리 속도 향상)

## 설치

### 1. Whisper 설치

```bash
pip install -U openai-whisper
```

### 2. FFmpeg 설치

Windows에서는 `winget`을 사용할 수 있음.

```bash
winget install Gyan.FFmpeg
```

설치 후 터미널을 다시 열고 정상적으로 인식되는지 확인:

```bash
ffmpeg -version
```

### 3. CUDA 사용 여부 확인

```bash
python testcuda.py
```

예시:

```text
PyTorch version : 2.14.0+cu130
CUDA build      : 13.0
CUDA available  : True
GPU count       : 1
```

`PyTorch version`에 `+cpu`가 표시되고 `CUDA available`이 `False`라면 CPU 전용 PyTorch가 설치된 상태임.

GPU를 사용하려면 환경에 맞는 CUDA 지원 PyTorch를 별도로 설치해야 함.

## 사용법

현재 디렉터리에 변환할 영상 파일을 넣은 뒤 스크립트를 실행함.

```bash
python lec2text.py
```

지원하는 영상 확장자:

* `.mp4`
* `.mkv`
* `.mov`
* `.avi`
* `.webm`

실행하면 현재 디렉터리에 있는 영상 파일 목록이 출력됨.

```text
변환할 영상 파일을 선택하세요:
1. lecture.mp4
2. interview.mp4

번호 입력: 1
```

이후 진행 상황 표시 방식을 선택할 수 있음.

```text
진행 표시 방식을 선택하세요:
1. 진행률 바 표시 (권장)
2. 인식된 세그먼트 텍스트를 콘솔에 출력
```

### 진행률 바

Whisper의 처리 진행률을 확인하면서 변환함.

긴 영상에서 현재 작업이 정상적으로 진행 중인지 확인하기 편함.

### 세그먼트 출력

Whisper가 인식한 내용을 처리 중 콘솔에 출력함.

```text
[00:00.000 --> 00:05.420] 안녕하세요.
[00:05.420 --> 00:11.830] 오늘은 운영체제에 대해서 알아보겠습니다.
```

## 동작 방식

1. 변환할 영상 파일 선택
2. CUDA 사용 가능 여부 확인
3. Whisper 모델 로드
4. 영상 파일을 Whisper로 직접 처리
5. 변환 결과 저장

별도의 MP3 파일이나 임시 세그먼트 파일은 생성하지 않음.

CUDA를 사용할 수 있으면 GPU를 사용하고, 사용할 수 없는 경우 CPU로 자동 전환됨.

## 출력

변환 결과는 다음 경로에 저장됨.

```text
result/영상파일명/
```

예를 들어:

```text
lecture.mp4
```

를 변환하면:

```text
result/
└─ lecture/
   ├─ lecture.txt
   ├─ lecture_segments.txt
   └─ lecture.srt
```

각 파일의 용도:

| 파일                     | 설명                      |
| ---------------------- | ----------------------- |
| `lecture.txt`          | 전체 음성을 일반 텍스트로 저장       |
| `lecture_segments.txt` | 각 음성 구간의 타임스탬프와 텍스트 저장  |
| `lecture.srt`          | 영상에서 사용할 수 있는 SRT 자막 파일 |

### 타임스탬프 텍스트 예시

```text
[1] 00:00:00 ~ 00:00:05
안녕하세요.

[2] 00:00:05 ~ 00:00:11
오늘은 Whisper에 대해서 알아보겠습니다.
```

## 기본 설정

현재 기본 Whisper 모델:

```python
MODEL_NAME = "medium"
```

기본 인식 언어:

```python
LANGUAGE = "ko"
```

필요한 경우 코드 상단의 설정값을 변경해서 사용할 수 있음.
