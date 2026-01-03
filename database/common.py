import geopandas as gpd
from shapely.geometry import Point, Polygon
BASE_SRID = 2450

def grid_from_point(p: Point, buffer_size: float = 0.5):
    poly = Polygon([
        (p.x - buffer_size, p.y - buffer_size),
        (p.x + buffer_size, p.y - buffer_size),
        (p.x + buffer_size, p.y + buffer_size),
        (p.x - buffer_size, p.y + buffer_size),
        (p.x - buffer_size, p.y - buffer_size)
    ])
    return poly


def get_wkt_from_shp(roi_shp):
    df = gpd.read_file(roi_shp)
    geom = df['geometry'].loc[0]
    return geom.wkt
