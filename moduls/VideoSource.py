# moduli/VideoSource.py
import time
from moduls.ThreadVideoCapture import ThreadedVideoCapture
from moduls.rtsp_stream import ThreadedRTSPStream


def get_video_source(args, input_path, rtsp_url):
    """
    Inicijalizuje i vraća odgovarajući video izvor (stream ili fajl) i prvi frejm.
    """
    if args.RTSP:
        # Koristimo 'source' kao generičko ime za video objekat
        source = ThreadedRTSPStream(rtsp_url)
        source.start()
        time.sleep(2.0)  # Sačekaj da se stream stabilizuje
        if not source.stream.isOpened():
            print(f"Greška: Ne mogu da otvorim RTSP stream na: {rtsp_url}")
            return None, None
    else:
        source = ThreadedVideoCapture(input_path)
        source.start()
        time.sleep(1.0)

    # Pročitaj prvi frejm da bi se inicijalizovali parametri (npr. dimenzije)
    grabbed, frame = source.read()
    if not grabbed:
        print("Greška: Ne mogu da pročitam prvi frejm sa video izvora.")
        if hasattr(source, 'stop'):
            source.stop()
        return None, None

    # ISPRAVKA: Vraćamo objekat 'source' i prvi frejm, a ne status 'grabbed'
    return source, frame