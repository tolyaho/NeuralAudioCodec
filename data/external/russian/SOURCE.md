# Russian test clips

Two read-speech utterances from the Russian (`ru_ru`) dev split of [FLEURS](https://huggingface.co/datasets/google/fleurs). They are used for the cross-language check in `notebooks/analysis.ipynb` and in the README audio table. The codec never saw Russian during training.

| File | FLEURS id | FLEURS audio file (`dev` split) | Speaker (FLEURS label) | Duration | Transcription (FLEURS `raw transcription`) |
|---|---|---|---|---|---|
| `fleurs_ru_1571.wav` | 1571 | `12522313282742242546.wav` | female | 10.20 s | Арабы также принесли в этот регион ислам, особенно он укоренился на Коморах и Майотте. |
| `fleurs_ru_1651.wav` | 1651 | `18402449285202103401.wav` | male | 10.56 s | Вариант, становящийся всё более популярным для тех, кто планирует взять академический год, это путешествовать и учиться. |

- **Source:** `data/ru_ru/audio/dev.tar.gz` and `data/ru_ru/dev.tsv` from <https://huggingface.co/datasets/google/fleurs>, downloaded 2026-10-29. The archive's SHA-256 is `4dc533cd0f312f749bd6d0fe968f3183288d938e07ed97dc141d9f10570457f9`.
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated on the FLEURS dataset card.
- **Attribution:** FLEURS, Google. A. Conneau, M. Ma, S. Khanuja, Y. Zhang, V. Axelrod, S. Dalmia, J. Riesa, C. Rivera, A. Bapna, "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", [arXiv:2205.12446](https://arxiv.org/abs/2205.12446), 2022.
- **Selection rule:** the first female and the first male utterance, in `dev.tsv` order, whose duration is between 9.5 s and 11 s.
- **Changes from the source:** the 32-bit float WAV files were converted to 16-bit PCM WAV (16 kHz mono, as published). The largest sample difference this introduces is 3.0e-5. Nothing was trimmed, resampled or level-adjusted.
