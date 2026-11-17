# Video review

Reviewed the 36 beat frames, contact sheets for all 18 scenes, phone-width frames and a representative scene seam after the final font/layout changes.

The film follows one shared experimental protocol, then explains each lab's mechanism and its most important limitation. A long code walkthrough and an exam-only sequence were considered; the interactive course supplies that extra depth instead.

P0 fixes completed: training/validation/test distinctions are explicit; diagram paths now connect their nodes; gradient descent has its own loss-valley diagram; numerical bars read the saved experiment results; Cyrillic fonts are bundled under OFL; no stock or fabricated screenshots. P1 fixes completed: smaller SVG labels raised to at least 36 design pixels; GSAP targets guarded so scenes do not emit missing-target warnings.

Final automated check: lint, runtime, layout, motion, contrast and craft all pass with zero errors and warnings. One intentional audio warning remains: the original background bed is extremely quiet. Narration is present in all 18 scenes; the bed is subordinate to speech. Keep that warning rather than making music compete with explanations.

Measured: timing from ffprobe for each narration MP3; 422.844-second root; automated browser/layout checks. Looked at: rendered-preview beat and cut frames, all chapter titles and educational diagrams. Audio will be measured in the exported MP4; it has not been listened to by a human. Sentence subtitle timings are approximate, not forced aligned.

Weakest shot: the backwards-pass network uses the same simplified topology as the forward pass, recoloured for gradients; a future longer tutorial could animate an explicit derivative along each edge. The current narration and accompanying course give the chain-rule equations.

The project is editable at http://127.0.0.1:3017/#project/video and through the HyperFrames desktop app.

Export verified: H.264 MP4, 1920×1080, 30 fps, 422.867 s, approximately 21 MB. Decoded frames at 30 s and 367 s inspected. Exported audio: stereo 48 kHz, integrated −17.7 LUFS, true peak −3.1 dB, no clipping and no analysis warnings. The audio was measured, not listened to. Offline browser checks confirm all 18 chapter buttons, 76 subtitle cues and time seeking.
