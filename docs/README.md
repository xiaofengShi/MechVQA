# MechVQA project website

Public URL: <https://xiaofengshi.github.io/MechVQA/>

## Preview and deployment

From the repository root:

```sh
python3 -m http.server 8793 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8793/>. No install or build step is required.
GitHub Pages uses **Deploy from a branch → main → /docs**. Push changes to
`main` to update the site. `.nojekyll` keeps the site static. Only `docs/` is
published; model files and training code are not copied into the site.

`index.html` contains all public content, `style.css` contains responsive and
print styles, and `script.js` progressively enhances example selection and
citation copying. Without JavaScript, all three examples remain available and
reference answers work through native details elements. No analytics, external
fonts, CDN libraries, or build dependencies are used. External research and
resource links need an internet connection; the downloaded site itself can be
read offline.

When editing, keep `citation.bib`, the inline BibTeX, `CITATION.cff`, and the
root README consistent. Check desktop and narrow screens, keyboard focus,
example selection, reference-answer disclosure, copy/download citation, local
assets, and fragment links. Scores on the site are a paper snapshot, not a live
leaderboard. Verify deployed content after Pages completes.

## Content provenance and editorial source

Verified against the repository and paper on 2026-09-22.

- Paper: `paper/2605.30794v1.pdf`, arXiv:2605.30794v1, 2026-05-29.
- Authors and affiliation markers follow the paper; citation follows the root
  README and `CITATION.cff`.
- Paper scale: about 3.3K drawings and 21K QA, ten tasks, three capabilities.
  These describe the paper dataset, not a claim that every record is public.
- Selected total scores from paper Table 2: MechVL-4B-RL 84.85;
  GLM-4.6V 78.91; Gemini-3-Pro-Preview 77.28; MechVL-4B-SFT 76.36;
  GPT-5 75.44; Qwen3-VL-4B-Instruct 60.23. Higher is better.
- MechVL-RL exceeds Gemini by 7.57 percentage points and GLM by 5.94.
  GLM is an open-source baseline; Gemini is the strongest closed-source
  baseline evaluated in the paper. MechVL is domain-adapted, while the
  general-purpose baselines are evaluated without domain adaptation.
- Total is the question-level accuracy aggregate. Capability columns in the
  root README are arithmetic means of the corresponding subtasks in Table 2,
  rounded to two decimals; they are not individual subtask scores.
- Training stages, paper Table 3: SFT 76.36; full-data DAPO 81.95;
  targeted DAPO 84.85. Table 3 is the source for these numbers; the older
  chart in paper Figure 2(c) has a different final label and is not reproduced.
- Public evaluation release: 1,185 QA and 562 unique images, from
  `benchmark_data/vqa_benchmark/mechvqa_benchmark.jsonl`.
- Public VQA-only SFT release: 12,749 train + 766 validation records and
  3,371 images, as documented in the root README. This has a different scope
  from the full paper dataset. Additional internal training data are excluded.
- ModelScope is the repository’s documented full-weight download destination.
  Hugging Face mirror availability must be checked on the respective cards.

## Assets and examples

`assets/overview.webp` is Figure 1 extracted from page 2 of the repository
paper (rendered crop: PDF coordinates 55,66–542,326 at 3×). It retains the
paper’s task taxonomy and example content. Figure attribution is visible on
site. No synthetic research figures are used.

The three example images are web-sized WebP conversions of these existing
public benchmark assets. Questions and reference answers are copied verbatim
from the corresponding JSONL records; no model inference is represented.

| Capability | Repository source image | Question |
|---|---|---|
| Recognition | `benchmark_data/images/1b/1b808e30b722895c46b88e0469c9fe40c0ddb079.png` | What is the value of the module m in the gear parameter table? |
| Reasoning | `benchmark_data/images/f2/f2b357ed93d1806bbcdb4abeb6acccb90577a6bb.jpg` | From the right-side sectional view, deduce the function of the serrations and spring, and explain how they achieve the locking function of the handle. |
| Judging | `benchmark_data/images/66/6601179687dd40bc7c9b519ec455f28639915eab.jpg` | Based on the table data, calculate the difference between s1 and s2 for d=20 and d=60, and determine whether the adhesive pad thickness varies with diameter. |

The hero reuses the Recognition example with a concise reference-answer label,
`m = 3`. Asset provenance and third-party drawing notices are retained; the
footer’s code-license link does not assert new ownership of source drawings.

## Design

White canvas, deep navy typography, muted green accents, restrained engineering
grid, system fonts, locally served research assets. The hierarchy prioritizes
the paper and dataset, followed by examples, results, training, resources, and
citation. Mobile layouts stack without hiding content; reduced motion and print
styles are included.
