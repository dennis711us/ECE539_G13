## Handwritten Study Notes Assistant<br><br>

Our project develops an AI notes-to-quiz assistant for university students that transforms a photograph
or scan of handwritten English lecture notes into structured Markdown study guides featuring organized
notes, a concise summary, review suggestions, and a self-test quiz. For example, an input note
containing “ReLU: max(0,x), problem: dying ReLU, solution? Leaky ReLU” is processed through our
pipeline to ultimately generate a short self-review quiz on activation functions and the dying ReLU
problem. We chose this direction because handwritten notes are fast and flexible during class but difficult
to search, reorganize, or review later; unlike standard OCR tools that only provide plain text transcription,
our system turns raw notes directly into actionable study material.<br><br>

## PaddleOCR — Open-source project:<br>
https://github.com/PaddlePaddle/PaddleOCR<br><br>

PaddleOCR is a widely used open-source OCR
project that parses text images into
machine-readable and structured content
JSON/Markdown. It can help university students
store their handwriting notes structurally and give raw
notes to LLM to help the LLM generate a notes
summary and related quiz questions for overviewing.
The above example shows how PaddleOCR is able to take images from any media and locate and
translate text. The field is the input image, while the labels are specific letters that the model can make
out from the image. This particular example fits the text-translation task, but it isn’t in English (our model
will focus on English characters).<br><br>

## Qwen3 — Related LLM technical resource: <br>
https://huggingface.co/Qwen/Qwen3-1.7B<br><br>

Qwen3-1.7B is an open-source lightweight large language model that can generate answers based on the
user's prompts. In our project, it will receive structured Markdown from PaddleOCR’s outputs and our
instructions in order to generate review questions and corresponding answers. Compared with
PaddleOCR, which focuses on extracting and structuring text from images, Qwen addresses the
downstream semantic task of understanding the recovered notes and transforming them into a useful
quiz. However, the model may generate incorrect or unsupported questions, so quiz faithfulness to the
source notes will need to be evaluated.<br><br>

## IAM Handwriting Database — Data source:<br>
https://fki.tic.heia-fr.ch/databases/iam-handwriting-database<br><br>

IAM is a large academic handwriting dataset containing scanned English handwriting together with
ground-truth transcriptions at page, line, and word levels. It can help us train our own handwriting
recognizing OCR. The main uncertainty is domain fit. IAM contains unconstrained English handwriting,
but much of its content is continuous prose rather than real university lecture notes. Student notes may
contain equations with abbreviations and mathematical signs, irregular layouts and incomplete phrases.
So we may need PaddleOCR to help recognize the academic parts and organize the spatial relationships.<br>
We inspected several IAM examples containing handwritten English lines paired with their ground-truth
transcriptions. Useful fields include the handwritten image itself, the corresponding text transcription, and
available page/line/word segmentation information. These labels fit the handwriting-recognition portion of
our task because a system output can be directly compared with the known transcription.