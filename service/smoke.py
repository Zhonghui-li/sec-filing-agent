"""What the image must satisfy before it is allowed to serve traffic.

Run INSIDE the built container. Needs no API key, no database and no money, because everything
here is a property of the installed dependency set rather than of the model's answers.

Both 2026-10-11 outages were invisible to every existing check and would have failed here:

  .text    `messages[-1].text` is a str in langchain-core 1.x and a METHOD in 0.3.x. The image
           was still building 0.3.x three days after the repo moved to 1.x, so every answer
           became "TypeError: expected string or bytes-like object, got 'method'". pytest never
           saw it: pytest runs against requirements.txt, the image installs
           service/requirements.txt, and nothing built the image.

  pyarrow  a clean resolve picked pyarrow 26, which requires numpy >= 2.0 against a pinned
           numpy 1.26.4. Nothing failed at startup — /health was fine and numeric questions
           answered from a baked JSON — and only narrative questions died, because they are the
           ones that reach edgartools. A liveness probe cannot catch that shape; importing the
           module can.

The rule for adding to this file: assert a CONTRACT the code depends on, not a version number.
Versions belong in requirements.txt, where tests/test_requirements_agree.py compares them.
"""
import sys


def check_agent_imports():
    """Every heavy import the request path makes, including the lazy ones.

    pyarrow and pandas arrive through edgartools and are imported only when a narrative question
    reaches the filings tools — far too late to find out they are incompatible.
    """
    import pyarrow                                     # noqa: F401  (first: the 10-11 culprit)
    import pandas                                      # noqa: F401
    import edgar                                       # noqa: F401  (edgartools)
    import agents.sec_agent                            # noqa: F401  (langchain, langgraph, tools)
    print("agent imports ok")


def check_service_imports():
    """The web layer. Fails outside the image, where fastapi is not installed — the dev venv
    builds from requirements.txt and the service deps live in service/requirements.txt."""
    import service.app                                 # noqa: F401  (fastapi, google-auth)
    print("service imports ok")


def check_message_text_is_a_string():
    """`.text` must BE a string, not something that becomes one when called."""
    from langchain_core.messages import AIMessage
    text = AIMessage(content="hello").text
    assert isinstance(text, str), (
        f"AIMessage.text is {type(text).__name__}, not a str — this is langchain-core 0.3.x "
        f"behaviour, where .text is a method. agents/sec_agent.py reads it as a property."
    )
    assert text == "hello", f"AIMessage.text returned {text!r}"
    print("AIMessage.text ok")


def check_responses_api_is_reachable():
    """The default path since a76c961. Construction only — no request, so no key and no cost."""
    import os
    from langchain_openai import ChatOpenAI
    os.environ.setdefault("OPENAI_API_KEY", "smoke-test-not-a-real-key")
    llm = ChatOpenAI(model="o4-mini", use_responses_api=True)
    assert llm.use_responses_api is True, "ChatOpenAI ignored use_responses_api"
    print("responses api ok")


def main():
    """Run every check and report every failure. Stopping at the first one hides the rest, and
    on 2026-10-11 the second problem was only found after the first was deployed away."""
    failed = []
    for check in (check_agent_imports, check_service_imports,
                  check_message_text_is_a_string, check_responses_api_is_reachable):
        try:
            check()
        except Exception as e:
            failed.append(f"{check.__name__}: {type(e).__name__}: {e}")
            print(f"FAILED {check.__name__}: {type(e).__name__}: {e}", file=sys.stderr)
    if failed:
        print(f"\nsmoke FAILED ({len(failed)} of 4)", file=sys.stderr)
        return 1
    print("smoke ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
