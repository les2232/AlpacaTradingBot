from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "assets" / "alpaca_trading_bot.ico"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    size = 256
    image = Image.new("RGBA", (size, size), (15, 23, 42, 255))
    draw = ImageDraw.Draw(image)

    # Warm gradient-style layered background.
    for i, color in enumerate(
        [
            (14, 116, 144, 255),
            (14, 165, 233, 220),
            (34, 197, 94, 190),
        ]
    ):
        inset = 16 + (i * 18)
        draw.rounded_rectangle(
            [inset, inset, size - inset, size - inset],
            radius=38,
            fill=color,
        )

    chart_points = [
        (52, 172),
        (92, 148),
        (126, 156),
        (160, 116),
        (204, 84),
    ]
    draw.line(chart_points, fill=(255, 255, 255, 255), width=16, joint="curve")
    draw.ellipse((188, 68, 220, 100), fill=(251, 191, 36, 255))

    font = ImageFont.load_default(size=36)
    draw.rounded_rectangle((38, 34, 126, 94), radius=18, fill=(255, 255, 255, 235))
    draw.text((56, 49), "AT", fill=(15, 23, 42, 255), font=font)

    image.save(
        OUTPUT_PATH,
        format="ICO",
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
