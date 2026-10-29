"""Reference analysis validates model output and chooses distinct source views offline."""
import json
import numpy as np
import pytest
from PIL import Image
from viewer.object_card import source_crops, parse_card


def test_card_validation():
    card = dict(name='bottle', confidence='low', visible_text='unknown', search_query='pump bottle',
                observations=['pump visible'], uncertainties=['brand unclear'], capture_next=['back view'])
    response = {'choices': [{'message': {'content': json.dumps(card)}}]}
    assert parse_card(response) == card
    card['confidence'] = 'certain'
    response['choices'][0]['message']['content'] = json.dumps(card)
    with pytest.raises(ValueError):
        parse_card(response)


def test_duplicate_camera_views_are_not_repeated(tmp_path):
    file = tmp_path/'frame.jpg'
    Image.new('RGB', (100, 100), 'white').save(file)
    points = np.random.default_rng(4).uniform([-.1, -.1, 1], [.1, .1, 1.1], (100, 3))
    frame = {'image': str(file), 'K': np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]]), 'c2w': np.eye(4)}
    crops = source_crops(points, dict(width=100, height=100, frames=[frame, frame]))
    assert len(crops) == 1
    assert crops[0][1].width > 8
