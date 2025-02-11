import logging
import pathlib
import time

import geopandas

import neatnet

start_time = time.time()

logging.basicConfig(
    filename="simplified_generation.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.NOTSET,
)
logger = logging.getLogger(__name__)

logging.info("")
logging.info("")
logging.info(" |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|")
logging.info(" | Generating Simplified Street Networks |")
logging.info(" |_______________________________________|")
logging.info("")
logging.info("")
logging.info("")

fua_city = {
    # 1133: "aleppo",
    # 869: "auckland",
    # 4617: "bucaramanga",
    # 809: "douala",
    # 1656: "liege",
    # 4881: "slc",
    8989: "wuhan",
}

# dict of cityname: fua ID
city_fua = {c: f for f, c in fua_city.items()}

for city, fua in city_fua.items():
    t1 = time.time()
    aoi = f"{city}_{fua}"

    logging.info("")
    logging.info("")
    logging.info("")
    logging.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ >>>>")
    logging.info("")
    logging.info("")
    logging.info(f"  ** {aoi} **")
    logging.info("")
    logging.info("")

    # input data
    original = geopandas.read_parquet(pathlib.Path(aoi, "original.parquet"))

    # output data
    simplified = neatnet.simplify_network(original)
    simplified.to_parquet(pathlib.Path(aoi, "simplified.parquet"))

    t2 = round((time.time() - t1) / 60.0, 2)

    logging.info("")
    logging.info("")
    logging.info(f"\t{aoi} runtime: {t2} minutes")
    logging.info("")
    logging.info("")
    logging.info("")
    logging.info("<<<< ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    logging.info("")

endtime_time = round((time.time() - start_time) / 60.0, 2)

logging.info("")
logging.info("")
logging.info(f"Total runtime: {endtime_time} minutes")
logging.info(
    "========================================================================="
)
logging.info("")
logging.info("")
