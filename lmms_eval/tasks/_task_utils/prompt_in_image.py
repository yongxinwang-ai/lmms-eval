import re
from typing import List

from PIL import Image, ImageDraw, ImageFont

MINIMAL_SOLVE_PROMPT = "Help me solve the problem"

_SIMPLE_LATEX_REPLACEMENTS = {
    r"\times": " * ",
    r"\cdot": " * ",
    r"\div": " / ",
    r"\pm": " +/- ",
    r"\mp": " -/+ ",
    r"\leq": " <= ",
    r"\le": " <= ",
    r"\geq": " >= ",
    r"\ge": " >= ",
    r"\neq": " != ",
    r"\approx": " ~= ",
    r"\sim": " ~ ",
    r"\infty": " infinity ",
    r"\pi": " pi ",
    r"\theta": " theta ",
    r"\alpha": " alpha ",
    r"\beta": " beta ",
    r"\gamma": " gamma ",
}


def _load_font(font_size: int) -> ImageFont.ImageFont:
    font_candidates = ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return max(0, bbox[2] - bbox[0])


def _split_long_token(draw: ImageDraw.ImageDraw, token: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    chunks = []
    current = ""
    for ch in token:
        candidate = f"{current}{ch}"
        if not current or _text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            chunks.append(current)
            current = ch
    if current:
        chunks.append(current)
    return chunks


def _wrap_text_to_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue

        words = paragraph.split()
        current = ""
        for word in words:
            if _text_width(draw, word, font) > max_width:
                long_parts = _split_long_token(draw, word, font, max_width)
            else:
                long_parts = [word]

            for part in long_parts:
                if not current:
                    current = part
                    continue

                candidate = f"{current} {part}"
                if _text_width(draw, candidate, font) <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = part

        if current:
            lines.append(current)
    return lines


def _build_text_panel(
    image_width: int,
    question_text: str,
    *,
    font_size: int = 28,
    padding: int = 20,
    line_spacing: int = 8,
    max_lines: int = 120,
    render_text: bool = True,
) -> Image.Image:
    normalized_question = normalize_question_text_for_image(question_text)
    if not normalized_question:
        return Image.new("RGB", (image_width, 0), "white")

    font = _load_font(font_size)
    measure_draw = ImageDraw.Draw(Image.new("RGB", (image_width, 1), "white"))
    max_text_width = max(1, image_width - 2 * padding)
    lines = _wrap_text_to_lines(measure_draw, normalized_question, font, max_text_width)

    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            suffix = " ..."
            while lines[-1] and _text_width(measure_draw, lines[-1] + suffix, font) > max_text_width:
                lines[-1] = lines[-1][:-1]
            lines[-1] = lines[-1] + suffix

    line_bbox = measure_draw.textbbox((0, 0), "Ag", font=font)
    line_height = max(1, line_bbox[3] - line_bbox[1]) + line_spacing
    panel_height = (padding * 2) + line_height * max(1, len(lines))

    panel = Image.new("RGB", (image_width, panel_height), "white")
    if render_text:
        draw = ImageDraw.Draw(panel)
        y = padding
        for line in lines:
            draw.text((padding, y), line, font=font, fill="black")
            y += line_height
    return panel


def _compose_panel_and_image(image_rgb: Image.Image, panel: Image.Image, panel_position: str = "top") -> Image.Image:
    composed = Image.new("RGB", (image_rgb.width, image_rgb.height + panel.height), "white")
    if panel_position == "bottom":
        composed.paste(image_rgb, (0, 0))
        composed.paste(panel, (0, image_rgb.height))
    else:
        composed.paste(panel, (0, 0))
        composed.paste(image_rgb, (0, panel.height))
    return composed


def normalize_question_text_for_image(text: str) -> str:
    if text is None:
        return ""

    normalized = str(text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("`", "")
    normalized = re.sub(r"<image\s*\d*>", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<image_\d+>", " ", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("\\\\", "\n")
    normalized = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", " ", normalized)
    normalized = normalized.replace("$$", " ").replace("$", " ")
    normalized = normalized.replace(r"\(", " ").replace(r"\)", " ")
    normalized = normalized.replace(r"\[", " ").replace(r"\]", " ")
    normalized = normalized.replace(r"\{", "{").replace(r"\}", "}")

    for _ in range(8):
        updated = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", normalized)
        if updated == normalized:
            break
        normalized = updated
    for _ in range(8):
        updated = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", normalized)
        if updated == normalized:
            break
        normalized = updated

    normalized = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\^\{([^{}]+)\}", r"^\1", normalized)
    normalized = re.sub(r"_\{([^{}]+)\}", r"_\1", normalized)
    normalized = normalized.replace(r"\left", " ").replace(r"\right", " ")
    normalized = normalized.replace(r"\,", " ").replace(r"\;", " ").replace(r"\:", " ")
    normalized = normalized.replace(r"\%", "%").replace(r"\#", "#")

    for latex_token, plain_token in _SIMPLE_LATEX_REPLACEMENTS.items():
        normalized = normalized.replace(latex_token, plain_token)

    # Keep unknown LaTeX command names (e.g., \sin -> sin) instead of dropping semantics.
    normalized = re.sub(r"\\([a-zA-Z]+)", r" \1 ", normalized)
    normalized = normalized.replace("{", " ").replace("}", " ")
    normalized = "".join(ch for ch in normalized if ch == "\n" or ch.isprintable())
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def render_question_on_image(
    image: Image.Image,
    question_text: str,
    *,
    font_size: int = 28,
    padding: int = 20,
    line_spacing: int = 8,
    max_lines: int = 120,
    panel_position: str = "top",
) -> Image.Image:
    if image is None:
        raise ValueError("image must not be None")

    image_rgb = image.convert("RGB")
    panel = _build_text_panel(
        image_rgb.width,
        question_text,
        font_size=font_size,
        padding=padding,
        line_spacing=line_spacing,
        max_lines=max_lines,
        render_text=True,
    )
    if panel.height == 0:
        return image_rgb
    return _compose_panel_and_image(image_rgb, panel, panel_position=panel_position)


def render_question_on_image_with_panel_crop(
    image: Image.Image,
    question_text: str,
    *,
    font_size: int = 28,
    padding: int = 20,
    line_spacing: int = 8,
    max_lines: int = 120,
    panel_position: str = "top",
) -> tuple[Image.Image, Image.Image]:
    if image is None:
        raise ValueError("image must not be None")

    image_rgb = image.convert("RGB")
    panel = _build_text_panel(
        image_rgb.width,
        question_text,
        font_size=font_size,
        padding=padding,
        line_spacing=line_spacing,
        max_lines=max_lines,
        render_text=True,
    )
    if panel.height == 0:
        return image_rgb, Image.new("RGB", (image_rgb.width, 1), "white")

    composed = _compose_panel_and_image(image_rgb, panel, panel_position=panel_position)
    crop = panel.copy() if panel_position == "top" else panel.copy()
    return composed, crop


def render_canvas_control_on_image(
    image: Image.Image,
    question_text: str,
    *,
    font_size: int = 28,
    padding: int = 20,
    line_spacing: int = 8,
    max_lines: int = 120,
    panel_position: str = "top",
) -> Image.Image:
    if image is None:
        raise ValueError("image must not be None")

    image_rgb = image.convert("RGB")
    panel = _build_text_panel(
        image_rgb.width,
        question_text,
        font_size=font_size,
        padding=padding,
        line_spacing=line_spacing,
        max_lines=max_lines,
        render_text=False,
    )
    if panel.height == 0:
        return image_rgb
    return _compose_panel_and_image(image_rgb, panel, panel_position=panel_position)
