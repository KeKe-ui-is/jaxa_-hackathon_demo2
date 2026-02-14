import calendar
import io
from datetime import date
from typing import Dict, Tuple

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw
from scipy.io import wavfile

JAXA_EARTH_API_URL = "https://data.earth.jaxa.jp/api/v1/observations"


def two_months_before_range(target_month: date) -> Tuple[str, str]:
    year = target_month.year
    month = target_month.month
    for _ in range(2):
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1

    _, last_day = calendar.monthrange(year, month)
    start = date(year, month, 1).isoformat()
    end = date(year, month, last_day).isoformat()
    return start, end


def normalize(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))


def fallback_satellite_values(lat: float, lon: float) -> Dict[str, float]:
    seed = int((abs(lat) * 1000) + (abs(lon) * 1000))
    rng = np.random.default_rng(seed)
    return {
        "ndvi": float(rng.uniform(0.1, 0.9)),
        "lst": float(rng.uniform(5.0, 40.0)),
        "precip": float(rng.uniform(20.0, 450.0)),
    }


def fetch_satellite_data(lat: float, lon: float, target_month: date) -> Dict[str, float]:
    start, end = two_months_before_range(target_month)

    params = {
        "lat": lat,
        "lon": lon,
        "start_date": start,
        "end_date": end,
        "variables": "ndvi_monthly,lst_monthly,precip_monthly",
    }

    try:
        res = requests.get(JAXA_EARTH_API_URL, params=params, timeout=20)
        res.raise_for_status()
        payload = res.json()

        # 想定レスポンス形式の違いに対応
        data = payload.get("data", payload)

        ndvi = float(data.get("ndvi_monthly", data.get("ndvi")))
        lst = float(data.get("lst_monthly", data.get("lst")))
        precip = float(data.get("precip_monthly", data.get("precip")))

        return {"ndvi": ndvi, "lst": lst, "precip": precip}
    except Exception as exc:  # API未接続時のフォールバック
        st.warning(
            "JAXA Earth API への接続に失敗したため、座標に基づく代替データを使います。"
            f"\n\n詳細: {exc}"
        )
        return fallback_satellite_values(lat, lon)


def create_space_landscape(ndvi: float, lst: float, precip: float) -> Image.Image:
    width, height = 320, 320

    ndvi_n = normalize(ndvi, 0.0, 1.0)
    lst_n = normalize(lst, -10.0, 45.0)
    precip_n = normalize(precip, 0.0, 500.0)

    base_brightness = int(70 + lst_n * 120)
    sky_top = (18 + int(28 * lst_n), 20 + int(30 * lst_n), 58 + int(80 * lst_n))
    sky_bottom = (255, 245, 170 + int(65 * lst_n))

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # 宇宙風グラデーション背景
    for y in range(height):
        t = y / max(1, height - 1)
        r = int(sky_top[0] * (1 - t) + sky_bottom[0] * t)
        g = int(sky_top[1] * (1 - t) + sky_bottom[1] * t)
        b = int(sky_top[2] * (1 - t) + sky_bottom[2] * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # 星
    rng = np.random.default_rng(int(ndvi_n * 1000 + precip_n * 2000 + lst_n * 3000))
    for _ in range(60):
        x = int(rng.uniform(0, width))
        y = int(rng.uniform(0, height * 0.45))
        size = int(rng.uniform(1, 3))
        color = (255, 255, int(rng.uniform(180, 255)))
        draw.ellipse((x, y, x + size, y + size), fill=color)

    # 地面
    ground_y = int(height * 0.52)
    ground_color = (102, 78 + int(75 * ndvi_n), 52)
    draw.rectangle((0, ground_y, width, height), fill=ground_color)

    # 水域（降水量依存）
    water_count = int(1 + precip_n * 3)
    for i in range(water_count):
        x1 = int((i / max(1, water_count)) * width * 0.75)
        x2 = int(x1 + width * (0.4 + precip_n * 0.2))
        y1 = int(ground_y + 35 + i * 16)
        y2 = y1 + int(20 + precip_n * 35)
        draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=(72, 142, 210))

    # 森（NDVI依存）
    tree_count = int(3 + ndvi_n * 16)
    for _ in range(tree_count):
        tx = int(rng.uniform(20, width - 20))
        ty = int(rng.uniform(ground_y - 40, height - 55))
        trunk_w = 4
        trunk_h = 12
        draw.rectangle((tx, ty, tx + trunk_w, ty + trunk_h), fill=(90, 50, 20))
        crown_r = int(8 + ndvi_n * 8)
        draw.ellipse(
            (tx - crown_r, ty - crown_r, tx + trunk_w + crown_r, ty + crown_r),
            fill=(30, 110 + int(ndvi_n * 120), 40),
        )

    # 明るい雰囲気（LST依存）
    if lst_n > 0.3:
        glow = Image.new("RGBA", (width, height), (255, 245, 180, int(55 * lst_n)))
        img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")

    # 建物（町らしさ）
    building_count = int(2 + (1 - ndvi_n) * 5)
    for _ in range(building_count):
        bw = int(rng.uniform(18, 38))
        bh = int(rng.uniform(35, 85))
        bx = int(rng.uniform(0, width - bw))
        by = ground_y - bh
        draw.rectangle((bx, by, bx + bw, ground_y), fill=(170, 168, 150))
        for wy in range(by + 6, ground_y - 6, 10):
            for wx in range(bx + 4, bx + bw - 4, 8):
                if rng.uniform() > 0.35:
                    draw.rectangle((wx, wy, wx + 3, wy + 4), fill=(255, 238, 170))

    # 惑星
    px, py = int(width * 0.8), int(height * 0.2)
    pr = int(22 + base_brightness * 0.07)
    draw = ImageDraw.Draw(img)
    draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=(255, 230, 130), outline=(255, 255, 255), width=3)

    return img


def synthesize_music(style: str, ndvi: float, lst: float, precip: float, duration_sec: int = 16) -> bytes:
    sr = 44100
    ndvi_n = normalize(ndvi, 0.0, 1.0)
    lst_n = normalize(lst, -10.0, 45.0)
    precip_n = normalize(precip, 0.0, 500.0)

    style_presets = {
        "クラシック": {"base_tempo": 74, "acoustic_boost": 1.15, "synth_boost": 0.65, "distortion": 0.02},
        "ロック": {"base_tempo": 96, "acoustic_boost": 0.95, "synth_boost": 0.85, "distortion": 0.12},
        "EDM": {"base_tempo": 118, "acoustic_boost": 0.7, "synth_boost": 1.2, "distortion": 0.08},
    }
    preset = style_presets[style]

    tempo = preset["base_tempo"] + int(lst_n * 72)
    beat_sec = 60.0 / tempo

    t = np.linspace(0, duration_sec, duration_sec * sr, endpoint=False)

    # アコースティック成分（NDVIが高いほど強い）
    acoustic = (
        0.6 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 330 * t)
        + 0.2 * np.sin(2 * np.pi * 440 * t)
    ) * (0.15 + 0.85 * ndvi_n) * preset["acoustic_boost"]

    # シンセ成分（降水量が高いほど強い）
    synth_freq = 110 + precip_n * 330
    synth = np.sign(np.sin(2 * np.pi * synth_freq * t)) * (0.15 + 0.85 * precip_n) * preset["synth_boost"]
    grit = np.tanh((acoustic + synth) * (1 + preset["distortion"]))

    # テンポに合わせたパーカッシブゲート
    gate = (np.sin(2 * np.pi * (1 / beat_sec) * t) > 0).astype(float)
    gate = 0.35 + 0.65 * gate

    # 雨ノイズ（降水量で増加）
    rng = np.random.default_rng(int((ndvi + lst + precip) * 1000))
    rain_noise = rng.normal(0, 1, len(t)) * 0.04 * precip_n

    audio = (grit + synth + rain_noise) * gate
    audio = audio / np.max(np.abs(audio) + 1e-8)
    audio_int16 = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    wavfile.write(buffer, sr, audio_int16)
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="JAXA Earth Soundscape", layout="centered")

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(160deg, #fff7ae 0%, #ffffff 55%, #ffef8f 100%);
        }
        .phone-frame {
            max-width: 390px;
            margin: 0 auto;
            border-radius: 36px;
            padding: 18px 16px;
            border: 3px solid #ffe261;
            box-shadow: 0 14px 30px rgba(0,0,0,0.18);
            background: rgba(255, 255, 255, 0.92);
        }
        .title {
            font-size: 1.55rem;
            font-weight: 800;
            color: #554100;
            margin-bottom: 0.4rem;
        }
        .sub {
            color: #7f6a00;
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }
        </style>
        <div class="phone-frame">
          <div class="title">🌍✨ JAXA Earth Soundscape</div>
          <div class="sub">座標と月を入力して、衛星データから画像と音楽を生成</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        lat = st.number_input("緯度 (Latitude)", min_value=-90.0, max_value=90.0, value=35.68, step=0.01)
        lon = st.number_input("経度 (Longitude)", min_value=-180.0, max_value=180.0, value=139.76, step=0.01)
        month = st.date_input("作成する月", value=date.today().replace(day=1))
        style = st.segmented_control("音楽スタイル", options=["クラシック", "ロック", "EDM"], default="クラシック")
        generate = st.button("画像と音楽を作成", type="primary", use_container_width=True)

    if generate and style:
        with st.spinner("衛星データを取得して作品を生成中..."):
            sat = fetch_satellite_data(lat, lon, month)
            image = create_space_landscape(sat["ndvi"], sat["lst"], sat["precip"])
            wav = synthesize_music(style, sat["ndvi"], sat["lst"], sat["precip"])

        st.success("生成が完了しました！")
        st.write(
            {
                "target_month": month.strftime("%Y-%m"),
                "used_two_months_before": two_months_before_range(month),
                "music_style": style,
                "ndvi_monthly": round(sat["ndvi"], 4),
                "lst_monthly_celsius": round(sat["lst"], 2),
                "precip_monthly_mm": round(sat["precip"], 2),
            }
        )

        st.image(image, caption="衛星データを反映した宇宙風ランドスケープ", use_container_width=True)
        st.audio(wav, format="audio/wav")


if __name__ == "__main__":
    main()
