"""
Cascaded voice AI agent on LiveKit.

Pipeline:  you speak  ->  STT (speech-to-text)  ->  LLM (thinks)  ->  TTS (speaks back)

All three models run on LiveKit Inference, so you only need your LiveKit keys in .env
(no OpenAI / Deepgram / Cartesia accounts required).

Run it and talk through your mic:
    uv run python agent.py console
"""

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, inference
from livekit.plugins import silero

load_dotenv()  # loads LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET from .env


class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a friendly voice assistant for a college workshop. "
                "Keep answers short and conversational — a sentence or two. "
                "Shorter replies also cost less, so don't ramble."
            )
        )


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        # --- The cascaded pipeline (all via LiveKit Inference, no extra keys) ---
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        # TTS is the biggest cost. deepgram/aura-2 ($30/M chars) is cheap + good.
        # Swap options if you like:
        #   "fish_audio/s2.1-pro-free"  -> free ($0), stretches your $2.50 for hours
        #   "cartesia/sonic-2"          -> most natural voice, pricier ($50/M chars)
        # If a model errors "not found", copy the exact id from your LiveKit
        # dashboard -> Agents -> Inference.
        tts=inference.TTS(model="deepgram/aura-2"),
        vad=silero.VAD.load(),  # detects when you start/stop talking (runs locally, free)
    )

    await session.start(room=ctx.room, agent=Assistant())
    await session.generate_reply(instructions="Greet the user and ask how you can help.")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
