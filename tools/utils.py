import re
import string
import numpy as np
from scipy.io import wavfile

def sanitize_filename(filename: str) -> str:
    # Define a set of valid characters
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)

    # Keep only valid characters
    sanitized_filename = ''.join(c for c in filename if c in valid_chars)

    # Replace multiple spaces with a single space
    sanitized_filename = re.sub(' +', ' ', sanitized_filename)

    return sanitized_filename


def save_wav(wav: np.ndarray, output_path: str, sample_rate=24000):
    # wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav_norm = wav * 32767
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))

def save_wav_norm(wav: np.ndarray, output_path: str, sample_rate=24000):
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))
    
def normalize_wav(wav_path: str) -> None:
    sample_rate, wav = wavfile.read(wav_path)
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wavfile.write(wav_path, sample_rate, wav_norm.astype(np.int16))

SUPPORT_VOICE = ['zu-ZA-ThembaNeural', 'zu-ZA-ThandoNeural',  'zh-TW-YunJheNeural', 'zh-TW-HsiaoYuNeural', 'zh-TW-HsiaoChenNeural', 'zh-HK-WanLungNeural', 
    'zh-HK-HiuMaanNeural', 'zh-HK-HiuGaaiNeural', 'zh-CN-shaanxi-XiaoniNeural', 'zh-CN-liaoning-XiaobeiNeural', 
    'zh-CN-YunyangNeural', 'zh-CN-YunxiaNeural', 'zh-CN-YunxiNeural', 'zh-CN-YunjianNeural', 
    'zh-CN-XiaoyiNeural', 'zh-CN-XiaoxiaoNeural', 'vi-VN-NamMinhNeural', 'vi-VN-HoaiMyNeural', 
    'uz-UZ-SardorNeural', 'uz-UZ-MadinaNeural', 'ur-PK-UzmaNeural', 'ur-PK-AsadNeural', 
    'ur-IN-SalmanNeural', 'ur-IN-GulNeural', 'uk-UA-PolinaNeural', 'uk-UA-OstapNeural', 
    'tr-TR-EmelNeural', 'tr-TR-AhmetNeural', 'th-TH-PremwadeeNeural', 'th-TH-NiwatNeural', 
    'te-IN-ShrutiNeural', 'te-IN-MohanNeural', 'ta-SG-VenbaNeural', 'ta-SG-AnbuNeural', 
    'ta-MY-SuryaNeural', 'ta-MY-KaniNeural', 'ta-LK-SaranyaNeural', 'ta-LK-KumarNeural', 
    'ta-IN-ValluvarNeural', 'ta-IN-PallaviNeural', 'sw-TZ-RehemaNeural', 'sw-TZ-DaudiNeural', 
    'sw-KE-ZuriNeural', 'sw-KE-RafikiNeural', 'sv-SE-SofieNeural', 'sv-SE-MattiasNeural', 
    'su-ID-TutiNeural', 'su-ID-JajangNeural', 'sr-RS-SophieNeural', 'sr-RS-NicholasNeural', 'sq-AL-IlirNeural', 'sq-AL-AnilaNeural', 
    'so-SO-UbaxNeural', 'so-SO-MuuseNeural', 'sl-SI-RokNeural', 'sl-SI-PetraNeural', 
    'sk-SK-ViktoriaNeural', 'sk-SK-LukasNeural', 'si-LK-ThiliniNeural', 'si-LK-SameeraNeural', 
    'ru-RU-SvetlanaNeural', 'ru-RU-DmitryNeural', 'ro-RO-EmilNeural', 'ro-RO-AlinaNeural', 'pt-PT-RaquelNeural', 'pt-PT-DuarteNeural', 'pt-BR-ThalitaNeural', 'pt-BR-FranciscaNeural', 
    'pt-BR-AntonioNeural', 'ps-AF-LatifaNeural', 'ps-AF-GulNawazNeural', 'pl-PL-ZofiaNeural', 
    'pl-PL-MarekNeural', 'nl-NL-MaartenNeural', 'nl-NL-FennaNeural', 'nl-NL-ColetteNeural', 
    'nl-BE-DenaNeural', 'nl-BE-ArnaudNeural', 'ne-NP-SagarNeural', 'ne-NP-HemkalaNeural', 
    'nb-NO-PernilleNeural', 'nb-NO-FinnNeural', 'my-MM-ThihaNeural', 'my-MM-NilarNeural', 
    'mt-MT-JosephNeural', 'mt-MT-GraceNeural', 'ms-MY-YasminNeural', 'ms-MY-OsmanNeural', 
    'mr-IN-ManoharNeural', 'mr-IN-AarohiNeural', 'mn-MN-YesuiNeural', 'mn-MN-BataaNeural', 
    'ml-IN-SobhanaNeural', 'ml-IN-MidhunNeural', 'mk-MK-MarijaNeural', 'mk-MK-AleksandarNeural', 
    'lv-LV-NilsNeural', 'lv-LV-EveritaNeural', 'lt-LT-OnaNeural', 'lt-LT-LeonasNeural', 'lo-LA-KeomanyNeural', 'lo-LA-ChanthavongNeural', 
    'ko-KR-SunHiNeural', 'ko-KR-InJoonNeural', 'ko-KR-HyunsuNeural', 'kn-IN-SapnaNeural', 
    'kn-IN-GaganNeural', 'km-KH-SreymomNeural', 'km-KH-PisethNeural', 'kk-KZ-DauletNeural', 
    'kk-KZ-AigulNeural', 'ka-GE-GiorgiNeural', 'ka-GE-EkaNeural', 'jv-ID-SitiNeural', 'jv-ID-DimasNeural', 
    'ja-JP-NanamiNeural', 'ja-JP-KeitaNeural', 'it-IT-IsabellaNeural', 'it-IT-GiuseppeNeural', 'it-IT-ElsaNeural', 
    'it-IT-DiegoNeural', 'is-IS-GunnarNeural', 'is-IS-GudrunNeural', 'id-ID-GadisNeural', 'id-ID-ArdiNeural', 
    'hu-HU-TamasNeural', 'hu-HU-NoemiNeural', 'hr-HR-SreckoNeural', 'hr-HR-GabrijelaNeural', 'hi-IN-SwaraNeural', 
    'hi-IN-MadhurNeural', 'he-IL-HilaNeural', 'he-IL-AvriNeural', 'gu-IN-NiranjanNeural', 'gu-IN-DhwaniNeural', 
    'gl-ES-SabelaNeural', 'gl-ES-RoiNeural', 'ga-IE-OrlaNeural', 'ga-IE-ColmNeural', 'fr-FR-VivienneMultilingualNeural', 
    'fr-FR-RemyMultilingualNeural', 'fr-FR-HenriNeural', 'fr-FR-EloiseNeural', 'fr-FR-DeniseNeural', 'fr-CH-FabriceNeural', 
    'fr-CH-ArianeNeural', 'fr-CA-ThierryNeural', 'fr-CA-SylvieNeural', 'fr-CA-JeanNeural', 'fr-CA-AntoineNeural', 
    'fr-BE-GerardNeural', 'fr-BE-CharlineNeural', 'fil-PH-BlessicaNeural', 'fil-PH-AngeloNeural', 'fi-FI-NooraNeural', 
    'fi-FI-HarriNeural', 'fa-IR-FaridNeural', 'fa-IR-DilaraNeural', 'et-EE-KertNeural', 'et-EE-AnuNeural', 
    'es-VE-SebastianNeural', 'es-VE-PaolaNeural', 'es-UY-ValentinaNeural', 'es-UY-MateoNeural', 'es-US-PalomaNeural', 
    'es-US-AlonsoNeural', 'es-SV-RodrigoNeural', 'es-SV-LorenaNeural', 'es-PY-TaniaNeural', 'es-PY-MarioNeural', 
    'es-PR-VictorNeural', 'es-PR-KarinaNeural', 'es-PE-CamilaNeural', 'es-PE-AlexNeural', 'es-PA-RobertoNeural', 
    'es-PA-MargaritaNeural', 'es-NI-YolandaNeural', 'es-NI-FedericoNeural', 'es-MX-JorgeNeural', 'es-MX-DaliaNeural', 
    'es-HN-KarlaNeural', 'es-HN-CarlosNeural', 'es-GT-MartaNeural', 'es-GT-AndresNeural', 'es-GQ-TeresaNeural', 
    'es-GQ-JavierNeural', 'es-ES-XimenaNeural', 'es-ES-ElviraNeural', 'es-ES-AlvaroNeural', 'es-EC-LuisNeural', 
    'es-EC-AndreaNeural', 'es-DO-RamonaNeural', 'es-DO-EmilioNeural', 'es-CU-ManuelNeural', 'es-CU-BelkysNeural', 
    'es-CR-MariaNeural', 'es-CR-JuanNeural', 'es-CO-SalomeNeural', 'es-CO-GonzaloNeural', 'es-CL-LorenzoNeural', 
    'es-CL-CatalinaNeural', 'es-BO-SofiaNeural', 'es-BO-MarceloNeural', 'es-AR-TomasNeural', 'es-AR-ElenaNeural', 
    'en-ZA-LukeNeural', 'en-ZA-LeahNeural', 'en-US-SteffanNeural', 'en-US-RogerNeural', 'en-US-MichelleNeural', 
    'en-US-JennyNeural', 'en-US-GuyNeural', 'en-US-EricNeural', 'en-US-EmmaNeural', 'en-US-ChristopherNeural', 
    'en-US-BrianNeural', 'en-US-AvaNeural', 'en-US-AriaNeural', 'en-US-AndrewNeural', 'en-US-AnaNeural', 
    'en-TZ-ImaniNeural', 'en-TZ-ElimuNeural', 'en-SG-WayneNeural', 'en-SG-LunaNeural', 'en-PH-RosaNeural', 
    'en-PH-JamesNeural', 'en-NZ-MollyNeural', 'en-NZ-MitchellNeural', 'en-NG-EzinneNeural', 
    'en-NG-AbeoNeural', 'en-KE-ChilembaNeural', 'en-KE-AsiliaNeural', 'en-IN-PrabhatNeural', 
    'en-IN-NeerjaNeural', 'en-IN-NeerjaExpressiveNeural', 'en-IE-EmilyNeural', 'en-IE-ConnorNeural', 
    'en-HK-YanNeural', 'en-HK-SamNeural', 'en-GB-ThomasNeural', 'en-GB-SoniaNeural', 'en-GB-RyanNeural', 
    'en-GB-MaisieNeural', 'en-GB-LibbyNeural', 'en-CA-LiamNeural', 'en-CA-ClaraNeural', 'en-AU-WilliamNeural', 
    'en-AU-NatashaNeural', 'el-GR-NestorasNeural', 'el-GR-AthinaNeural', 'de-DE-SeraphinaMultilingualNeural', 
    'de-DE-KillianNeural', 'de-DE-KatjaNeural', 'de-DE-FlorianMultilingualNeural', 'de-DE-ConradNeural', 
    'de-DE-AmalaNeural', 'de-CH-LeniNeural', 'de-CH-JanNeural', 'de-AT-JonasNeural', 'de-AT-IngridNeural', 
    'da-DK-JeppeNeural', 'da-DK-ChristelNeural', 'cy-GB-NiaNeural', 'cy-GB-AledNeural', 'cs-CZ-VlastaNeural',
    'cs-CZ-AntoninNeural', 'ca-ES-JoanaNeural', 'ca-ES-EnricNeural', 'bs-BA-VesnaNeural', 'bs-BA-GoranNeural', 
    'bn-IN-TanishaaNeural', 'bn-IN-BashkarNeural', 'bn-BD-PradeepNeural', 'bn-BD-NabanitaNeural', 'bg-BG-KalinaNeural', 
    'bg-BG-BorislavNeural', 'az-AZ-BanuNeural', 'az-AZ-BabekNeural', 'ar-YE-SalehNeural', 'ar-YE-MaryamNeural', 
    'ar-TN-ReemNeural', 'ar-TN-HediNeural', 'ar-SY-LaithNeural', 'ar-SY-AmanyNeural', 'ar-SA-ZariyahNeural', 
    'ar-SA-HamedNeural', 'ar-QA-MoazNeural', 'ar-QA-AmalNeural', 'ar-OM-AyshaNeural', 'ar-OM-AbdullahNeural', 
    'ar-MA-MounaNeural', 'ar-MA-JamalNeural', 'ar-LY-OmarNeural', 'ar-LY-ImanNeural', 'ar-LB-RamiNeural', 'ar-LB-LaylaNeural', 
    'ar-KW-NouraNeural', 'ar-KW-FahedNeural', 'ar-JO-TaimNeural', 'ar-JO-SanaNeural', 'ar-IQ-RanaNeural', 'ar-IQ-BasselNeural', 
    'ar-EG-ShakirNeural', 'ar-EG-SalmaNeural', 'ar-DZ-IsmaelNeural', 'ar-DZ-AminaNeural', 'ar-BH-LailaNeural', 'ar-BH-AliNeural', 
    'ar-AE-HamdanNeural', 'ar-AE-FatimaNeural', 'am-ET-MekdesNeural', 'am-ET-AmehaNeural', 'af-ZA-WillemNeural', 'af-ZA-AdriNeural']