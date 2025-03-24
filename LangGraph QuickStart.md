https://langchain-ai.github.io/langgraph/tutorials/introduction/

# LangGraph QuickStart
이 튜토리얼에서는, 랭그래프를 이용하여 고객 지원 챗봇을 만들어 볼 것이며, 이 챗봇은 아래와 같은 기능을 갖고 있습니다.

✅ 웹 검색으로 일반적인 질문에 답하고

✅ 여러 호출(요청) 간에도 대화 상태를 유지

✅ 복잡한 질문은 사람에게 전달하여 검토

✅ custom state를 활용해 동작을 제어

✅ 이전 대화를 되돌리고 다른 경로로 탐색 가능

기본적인 챗봇부터 시작해, 점차 더 정교한 기능들을 추가해 나갈 것입니다. 이 과정에서 LangGraph의 핵심 개념들도 함께 소개할 예정입니다. 그럼 시작해볼까요? 🌟

## Setup
첫번째, 필요한 패키지들을 설치하고 환경을 설정하세요.
``` shell
capture --no-stderr
pip install -U langgraph langsmith langchain_anthropic
```
- langgraph: LangChain 기반의 multiStep workflow를 그래프 구조로 구성할 수 있게 해주는 라이브러리
- langsmith: LangChain, LangGraph 애플리케이션의 디버깅, 모니터링, 실험 관리를 위한 도구
- langchain_anthropic: LangChain에서 Anthropic 모델(Claude 등)을 사용할 수 있게 해주는 패키지

``` python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
```

> LangGraph 개발을 위해 LangSmith 설정하기
>  LangSmith는 LangGraph로 만든 LLM 애플리케이션의 트레이스 데이터를 활용해 디버깅, 테스트, 모니터링할 수 있도록 도와줍니다. 
> https://docs.smith.langchain.com/