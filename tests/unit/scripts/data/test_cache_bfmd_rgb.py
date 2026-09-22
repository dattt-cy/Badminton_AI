from scripts.data_preparation.cache_bfmd_rgb import crop_from_percent, nearest_box


def test_nearest_box_and_percent_crop():
    track={100:(25.0,25.0,10.0,20.0),104:(30.0,30.0,10.0,20.0)}
    assert nearest_box(track,102)==track[100]
    assert nearest_box(track,120) is None
    x1,y1,x2,y2=crop_from_percent(track[100],width=1000,height=500,padding=.0)
    assert 0<=x1<x2<=1000
    assert 0<=y1<y2<=500
