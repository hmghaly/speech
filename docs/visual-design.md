# Visual design, company branding

The first pass used a generic gray/white panel layout and did not meet the requested visual standard. This version replaces that direction across the listing, creation, speaker and listener pages.

## Direction

A language research workspace with an audible identity: the original gold/brown company logo on white navigation, a brown brand surface and one static speech waveform on the home page. Keep participant tasks quiet and immediate. Use gold for primary actions and confirmed progress, rather than adding gold borders to every surface.

The home page pairs a substantial Chinese headline with the speech graphic. Project creation leads with the sentence file, then essential project settings; extra metadata uses native disclosure. Recording and listening pair short guidance with a focused task panel. Completion has its own clear visual treatment. Storage context appears beneath the work.

## Tokens

| Role           | Value     |
| -------------- | --------- |
| Canvas         | `#F8F6F2` |
| Surface        | `#FFFFFF` |
| Text           | `#30271F` |
| Brand brown    | `#493524` |
| Secondary text | `#6B6055` |
| Action gold    | `#F0BD4F` |
| Gold text      | `#76531A` |
| Gold tint      | `#FBF1DC` |
| Border         | `#E4DCD1` |
| Control border | `#998773` |
| Error          | `#A3303B` |
| Keyboard focus | `#82601F` |

Use dark text on gold, white text on brand brown, and muted text on white/light canvas. Borders on controls remain darker than decorative dividers. Recording uses red plus explicit text and a stop symbol.

Typography uses the Chinese/system sans stack for interface and research content. The home headline uses the system Chinese serif stack where available (Noto Serif CJK SC, Songti, STSong, SimSun, then Georgia/serif). No font download is required. Home type scales from 48px on mobile to 66px on desktop; task headings use 30–42px, research sentences 28–36px with 1.8 line height. Do not apply display serif styling to research stimuli.

Navbar labels use title case consistently, including browser-translated text. The logo has a small optical offset to center its visible artwork, accounting for the original image’s transparent margins.

## Layout and behavior

- Maximum shell width 1240px; desktop gutters 36px, phone gutters 20px plus safe areas.
- Task layout: 270px guidance column plus flexible content, 54px gap. At 960px the column/gap tighten; below 720px tasks stack and secondary guidance condenses.
- Primary actions at least 48px high, recording 58px, option rows 58px.
- Surface corners 18px, controls 9–10px, speech graphic 24px with one larger corner. No uniform shadow/card kit.
- Research controls, progress, saving and question transitions are immediate. No waveform movement or decorative entrance animations.
- Hover changes apply only with hover capability and a fine pointer. Keyboard focus remains visible; timer updates are not live announcements.

## Static setup

`assets/css/tokens.css` contains plain CSS variables. `assets/css/app.css` contains ordinary layered CSS; component rules sit above Tailwind's base reset. Each HTML page loads the official Tailwind browser CDN and uses utilities for small spacing adjustments. There is no build process or dependency manifest.

## Review status

The original logo and all four page shells were checked in the in-app browser. Desktop and phone-width layouts were inspected; physical microphone, real-phone behavior and Safari/Firefox still require verification. The layout follows the second pass approved by the user; the user has accepted the warm company branding.

## Skill decisions

Frontend design guidance keeps the speech graphic as the main visual gesture and research content neutral. Mobile guidance preserves safe-area padding, 16px inputs, visible keyboard focus and pointer-gated hover styles. Motion guidance rejects entrance and trial-transition animations for these repeated tasks; controls respond immediately.

## Error presentation

Errors use plain tinted surfaces and readable text, without an accent border. Project validation appears beside the affected field; save failures sit beside Save. Errors receive focus and scroll into view when necessary, and editing clears the related message. Hidden optional fields open when invalid.

## Anti-ui-slop audit — 10 October 2026

Scope: project listing, creation, speaker registration and listener entry rendered at 1280px and 390px. Source review also covered recording/playback controls, loading, empty, disabled, retry and completion states. No clipping or horizontal overflow was found in the rendered screens. The accepted logo, warm palette, speech graphic and task hierarchy stay in place. The requested error accent border has been removed across all pages.

The three findings are now corrected:

1. Participant errors receive focus and scroll into view. At 390px the listener’s missing-choice error was fully visible (395–448px in an 844px viewport), with no horizontal overflow.
2. Invalid file parsing retains its specific failure through Save; selecting a valid replacement restores the sentence preview.
3. Enabled playback buttons use a dark brown pressed background (`#352719`) with white text. Browser computed styles confirmed this state; physical touch verification remains pending.

## Internationalization

All authored interface text has explicit, stable DOM IDs. `assets/i18n/messages.json` uses these IDs as the keys in `messages`, with both `zh-Hans` and `en` values. Every root HTML page starts with `lang="zh-Hans"`; set it to `en` when deploying the English interface. No language selector is required for the current scope.

For example, `file-help` selects `messages["file-help"]["zh-Hans"]`. Changing controls keep their DOM IDs: `save-project` contains a language-specific map of `create.save` and `create.saving` states. `data-i18n` always equals the element’s ID; `data-i18n-state` selects its current wording. Placeholders and accessible labels use the same ID lookup.

Repeated project and preview rows use the JSON `templates` entries, such as `ui-projects-{id}-count`, expanded with the stored item ID. This gives each generated text element a stable ID without predeclaring future project IDs. Imported sentences, project titles and study questions retain their own text and IDs; translating interface copy never changes research stimuli. Browser-owned file picker text is outside this dictionary.

The translation test checks ID uniqueness, ID/key correspondence, default Chinese and both language entries on all four pages. Browser checks covered English creation, Chinese project creation and speaker entry, listener validation and audio loading/playback. Actual microphone capture and Safari/Firefox/real-phone verification remain pending.
