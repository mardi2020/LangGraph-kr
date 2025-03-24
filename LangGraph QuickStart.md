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

## Part 1: 기본적인 챗봇 만들기
먼저, LangGraph를 이용하여 간단한 챗봇을 만들어봅시다. 이 챗봇은 사용자 메시지를 바로 응답합니다. 간단하지만, LangGraph로 구축할 때의 핵심 개념들을 잘 보여줍니다. 이 섹션이 끝날 때, 기본적인 챗봇을 완성하게 될 것입니다.

`StateGraph`를 만들어 봅시다. `StateGraph` 객체는 챗봇 구조를 "상태 머신(state machine)"으로 정의합니다. 여기에 LLM과 챗봇이 호출할 수 있는 함수들을 노드(node)로 추가하고 함수들 간의 전환 방식으로 엣지(edge)로 지정합니다.
``` python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # messages는 "list" 타입입니다. 주석에 있는 `add_messages` 함수는  
    # 이 state key가 어떻게 갱신되어야 하는지를 정의합니다.  
    # (이 경우, 기존 메시지를 덮어쓰지 않고 리스트에 메시지를 추가합니다)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
```
API Reference: [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph) | [START](https://langchain-ai.github.io/langgraph/reference/constants/#langgraph.constants.START) | [END](https://langchain-ai.github.io/langgraph/reference/constants/#langgraph.constants.END) | [add_messages](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages)

그래프는 두개의 key task를 다룰 수 있게 되었습니다:
1. 각 node는 현재 state를 입력으로 받아서 그에 대한 업데이트 결과를 상태에 반영하여 출력할 수 있습니다.
2. Annotated 문법과 함께 사용하는 미리 정의된 add_messages 함수 덕분에 메시지에 대한 갱신은 기존 리스트를 덮어쓰지 않고 새 항목을 추가하는 방식으로 동작합니다.

> Concept
> 그래프를 정의할 때, 첫번째 단계는 `State`를 정의하는 것입니다. `State`는 그래프의 스키마와 상태 갱신을 처리하는 reducer 함수들을 포함합니다. 여기 예제에서는 `State`가 `messages`라는 하나의 키만 가진 `TypedDict`로 정의되어 있습니다. `add_messages` reducer 함수는 새로운 메시지를 리스트에 덮어씌우지 않고 추가할 때 사용합니다. reducer annotation없는 Keys는 이전 값들로 덮어씌우게 됩니다. state, reducers, 그리고 관련 개념들에 대해 더 알고 싶다면 이 [가이드](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages)를 참고하세요.

그 다음, "chatbot" 노드를 추가합니다. 노드들은 work의 유닛들로 나타낼 수 있으며 일반적인 파이썬 함수입니다.
``` python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# 첫번째 인자는 유니크한 노드의 이름입니다.
# 두번쨰 인자는 노드가 사용될 때마다 호출될 함수 또는 객체입니다.
graph_builder.add_node("chatbot", chatbot)
```
API Reference: [ChatAnthropic](https://python.langchain.com/api_reference/anthropic/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html)

**Notice** `chatbot` 노드 함수가 현재 state를 입력받고 "messages" 키에 업데이트된 메시지 리스트를 포함한 딕셔너리를 반환하는 방식에 주목하세요. 이것이 모든 LangGraph node 함수의 기본 패턴입니다.

`State`의 `add_messages` 함수는 llm의 응답 메시지를 현재 상태에 이미 존재하는 메시지 리스트에 추가합니다.

그 다음, `entry` 지점을 추가해 봅시다. 이것은 그래프가 실행될 떄마다 어디서부터 작업을 시작해야할지를 알려줍니다.
```python
graph_builder.add_edge(START, "chatbot")
```

유사하게, `finish` 지점도 지정해 봅시다. 그래프에게 "이 노드가 실행될 때마다 여기서 종료한다"라고 지시하는 의미입니다.
``` python
graph_builder.add_edge("chatbot", END)
```

마지막으로, 그래프를 실행시킬 수 있게 하기 위해, "`compile()`"을 graph builder에서 호출하면 됩니다. 이렇게 하면 현재 상태에 대해 실행할 수 있는 "`CompiledGraph`" 을 생성하게 됩니다.
``` python
graph = graph_builder.compile()
```

그래프는 `get_graph` 메서드와 `draw_ascii`, `draw_png`와 같은 "draw" 메서드 하나를 사용해 시각화할 수 있습니다. (각 draw 메서드는 추가적인 의존성 패키지 설치가 필요합니다.) 
``` python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```
