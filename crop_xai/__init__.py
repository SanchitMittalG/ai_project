"""Leakage-safe reproduction of the crop recommendation paper."""

FEATURES = ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")
TARGET = "label"

PAPER_CROPS = (
    "apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton",
    "grapes", "jute", "kidneybeans", "lentil", "maize", "mango", "mothbeans",
    "mungbean", "muskmelon", "orange", "papaya", "pigeonpeas", "pomegranate",
    "rice", "watermelon",
)
