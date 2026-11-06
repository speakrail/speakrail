# Platforms

One folder per hardware family. `platform.toml` holds the memory rules; `detect.py` has a `detect()` that returns what it finds, or None if this isn't the machine. To add a family, add a folder; nothing in core changes.

`reserve_gib` is memory kept free for the OS and other apps. A config fits if its measured peak plus the reserve fits in the machine's memory.
