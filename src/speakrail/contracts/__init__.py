"""The contracts between the harness and its engines. An engine fills one role and declares what it can do; the
conformance checks hold it to that. Bump CONTRACT_VERSION when a change would break existing engines."""

CONTRACT_VERSION = 1
ROLES = ("asr", "tts", "llm")                # also the start order: the LLM claims what GPU memory is left

REQUIRED = {"llm": set(), "asr": {"turn_head"}, "tts": set()}

CAPABILITIES = {
    "llm": {
        "chat": "POST /v1/chat/completions, used by the listening notes",
        "parallel": "several requests make progress at once; needed for notes and speculation",
        "per_request_adapter": "the base model and the LoRA are chosen per request by `model`",
        "cancel": "closing a stream stops the work",
        "prefix_cache": "a request sharing a prefix with an earlier one is not prefilled again",
    },
    "asr": {
        "transcribe": "timed words from speech",
        "turn_head": "`turn` events every 80 ms. Required: the harness ends turns from these frames, silence fallback included. "
                     "An engine without a real head can send them from a simple VAD",
        "anticipation": "p[5:8] of the turn head hold the anticipation outputs (ANT4)",
        "peek": "drain the delay line on a snapshot of the decoder",
    },
    "tts": {
        "stream": "PCM arrives while it is being generated",
        "cancel": "closing the connection stops generation",
        "clone": "voice_ref: a voice sent with the request",
    },
}
