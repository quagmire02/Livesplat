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
    assert parse_card(response) == dict(card, scan_issues=[], category='', repair_advice='none')
    grounded = dict(card, category='container', scan_issues=['view 1: table bits at the base'],
                    repair_advice='remove_attachments')                 # SAM-grounded prompt fields
    response['choices'][0]['message']['content'] = json.dumps(grounded)
    assert parse_card(response) == grounded
    grounded['repair_advice'] = 'reshape from a product photo'          # not an allowed action
    response['choices'][0]['message']['content'] = json.dumps(grounded)
    assert parse_card(response)['repair_advice'] == 'none'
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
