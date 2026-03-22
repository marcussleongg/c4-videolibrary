"""Prompts for body-worn camera video analysis (shared by pipeline + notebooks)."""

# Combined scene + transcript + prosody in one model call.
COMBINED_PROMPT = """\
You are producing metadata for full-text and semantic search over body-worn camera segments. \
All timestamps are relative to this clip's start (0:00).

Output ONLY these three sections, in this order, with no preamble or closing commentary.

=== SCENE DESCRIPTION ===
Write dense, non-repetitive prose. Cover each bullet once; do not restate the same facts in multiple places.

- EVENTS: What happens and when. For major beats, prefix with [MM:SS] in clip time \
(e.g. [00:05] Officer approaches the driver's window). Describe actions as they unfold.
- PEOPLE: Every person visible — clothing colors/types, actions, position, approximate age/build.
- VEHICLES: Type, color, make/model if identifiable, notable features.
- OBJECTS: Everything visible, including small or background items.
- ENVIRONMENT: Indoor/outdoor, lighting, time of day, weather, location type.
- VISIBLE TEXT: Signs, plates, badges, screen text. If unreadable or occluded, say so; \
do not invent characters or numbers.

Whenever the footage shows something a user might search for — clothing colors, lighting/time of day, \
vehicles, people, actions — use literal, searchable wording (e.g. red shirt, night, vehicle pulled over).

Body-worn audio often includes radio bleed, wind, traffic, and muffled speech. In the scene section, \
you may note obvious audio context briefly if it affects interpretation; full speech goes under TRANSCRIPT.

After the prose, add exactly one line of the form:
TIME_OF_DAY: <value>
where <value> is exactly one of: day, night, twilight.
Infer primarily from lighting and the scene; do not guess from audio alone.

=== TRANSCRIPT ===
Transcribe ALL speech verbatim with timestamps relative to this clip (starting at 0:00).
Format: [MM:SS] Speaker: "words spoken"
If multiple speakers, label them (Officer, Driver, Bystander, etc. or Speaker 1, Speaker 2).
Use [unclear] for unintelligible stretches.
If no speech is audible, write "No speech detected."
Include prominent non-speech in brackets if it is something that is notable, e.g. [explosion], [gunshot], [crash].

=== PROSODY ===
Characterize the human speech in this clip. Ignore background noise — that belongs in SCENE DESCRIPTION.
Return ONLY a single JSON object — no markdown fences, no text before or after.

{
  "max_volume": <"quiet", "normal", "loud", or "shouting" — peak human voice level>,
  "emotional_tones": [<one or more short lowercase labels, e.g. "calm", "tense", "agitated", "distressed", "angry">]
}

If no speech is present, use: {"max_volume": "quiet", "emotional_tones": []}
Do not put time_of_day in this JSON; it is set in TIME_OF_DAY under SCENE DESCRIPTION."""

# Scene-only path (swapped providers or describe() only).
SCENE_ONLY_PROMPT = """\
You are producing metadata for full-text and semantic search over body-worn camera segments. \
All timestamps are relative to this clip's start (0:00).

Describe everything visible in dense, non-repetitive prose, in temporal order.

- EVENTS: What happens and when. For major beats, prefix with [MM:SS] in clip time. \
Describe actions as they unfold.
- PEOPLE: Every person visible — clothing colors/types, actions, position, approximate age/build.
- VEHICLES: Type, color, make/model if identifiable, notable features.
- OBJECTS: Everything visible, including small or background items.
- ENVIRONMENT: Indoor/outdoor, lighting, time of day, weather, location type.
- VISIBLE TEXT: Signs, plates, badges. If unreadable, say so; do not invent text.

Use literal, searchable wording for anything a user might query (clothing colors, night/day, vehicles, actions).

On its own line after the description (required):
TIME_OF_DAY: one of day, night, twilight

Then list key events, one per line (optional [MM:SS] prefix):
EVENT: [00:12] officer approaches vehicle
EVENT: driver exits car"""

TRANSCRIPT_ONLY_PROMPT = """\
Transcribe ALL speech verbatim with timestamps relative to this clip (starting at 0:00).
Format: [MM:SS] Speaker: "words spoken"
If multiple speakers, label them (Officer, Driver, Bystander, etc. or Speaker 1, Speaker 2).
Use [unclear] for unintelligible stretches.
If no speech is audible, write "No speech detected."
Include prominent non-speech in brackets if it is something that is notable, e.g. [explosion], [gunshot], [crash].
"""

PROSODY_ONLY_PROMPT = """\
Characterize the volume and emotional tone of human speech in this body-worn camera footage. \
Ignore background noise (sirens, engines, wind, radio) — focus only on voices.

Return ONLY a single JSON object — no markdown fences, no text before or after.

{
  "max_volume": <"quiet", "normal", "loud", or "shouting" — peak human voice level>,
  "emotional_tones": [<one or more short lowercase labels, e.g. "calm", "tense", "agitated", "distressed", "angry">]
}

If no speech is present, use: {"max_volume": "quiet", "emotional_tones": []}
"""
