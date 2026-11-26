import datetime
import json
import random
import re
import urllib.request
from enum import Flag, auto
from pathlib import Path

import discord
import fugashi
from transformers import pipeline

import settings

# --- Sentiment Analysis Model (Japanese BERT) ---
MODEL_NAME = "llm-book/bert-base-japanese-v3-wrime-sentiment"
sentiment_pipeline = pipeline(model=MODEL_NAME)

# --- Morphological Analyzer ---
tagger = fugashi.Tagger()

JST = datetime.timezone(datetime.timedelta(hours=9))
RULE_FILE = Path(__file__).parent / "kw_matching_rule.txt"


class Word:
    def __init__(self, word):
        self.text = word.surface
        if any(pos in word.pos for pos in ("動詞", "形容詞", "形容動詞", "助動詞")):
            self.basicForm = word.feature.lemma
        else:
            self.basicForm = word.surface
        self.pos = word.pos

    def wordInfo(self):
        return f"{self.text}\t{self.basicForm}\t{self.pos}"


def fugashiAnalyzer(text):
    return [Word(w) for w in tagger(text)]


# --- Dialogue Strategies (Non-Task-Oriented) ---
kRuleList = []
candidateList = []


class KeywordMatchingRule:
    def __init__(self, keyword, response):
        self.keyword = keyword
        self.response = response


class ResponseCandidate:
    def __init__(self, response, score):
        self.response = response
        self.score = score

    def print(self):
        print(f"候補文 [{self.response}, {self.score:.5f}]")


def setupKeywordMatchingRule():
    kRuleList.clear()
    with open(RULE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            keyword, response = line.split(",", 1)
            kRuleList.append(KeywordMatchingRule(keyword.strip(), response.strip()))


def generateResponseByRule(inputText):
    for rule in kRuleList:
        if rule.keyword in inputText:
            cdd = ResponseCandidate(rule.response, 1.0 + random.random())
            candidateList.append(cdd)


def generateResponseByInputTopic(inputWordList):
    textList = ["は好きですか？", "って何ですか？"]
    for w in inputWordList:
        pos2 = w.pos.split(",")
        if pos2[0] == "名詞":
            cdd = ResponseCandidate(
                w.basicForm + random.choice(textList),
                0.7 + random.random(),
            )
            candidateList.append(cdd)


def generateOtherResponse():
    bunanList = ["なるほど", "それで？"]
    cdd = ResponseCandidate(random.choice(bunanList), 0.5 + random.random())
    candidateList.append(cdd)


def generateNegaposiResponse(text):
    model_prediction = sentiment_pipeline(text)
    output = "よかったね" if model_prediction[0]["label"] == "positive" else "ざんねん"
    cdd = ResponseCandidate(output, 0.7 + random.random())
    candidateList.append(cdd)


# --- State Management (Task-Oriented Modes) ---
class State(Flag):
    DEFAULT = auto()
    JANKEN = auto()
    DAJARE = auto()
    ZIPCODE = auto()
    FORMULA1 = auto()


state = State.DEFAULT


def janken(inputText):
    hand = ["グー", "チョキ", "パー"]
    prompt = "グー／チョキ／パーを入力（ENDで終了）"
    if inputText in hand:
        return f"あなた：{inputText} 私：{random.choice(hand)}。{prompt}"
    return prompt


def dajare(inputText):
    dajare_dic = {
        "ふとん": "ふとんがふっとんだ",
        "言い訳": "言い訳していいわけ？",
        "トイレ": "トイレ行っといれ",
        "あさり": "あっさりあさり味",
        "スキー": "スキーが好きー",
    }
    return dajare_dic.get(inputText, "駄洒落が言えないよ~")


def getAddressFromZip(zipcode):
    url = f"https://zipcloud.ibsnet.co.jp/api/search?zipcode={zipcode}"
    try:
        with urllib.request.urlopen(url, timeout=5) as res:
            data = json.loads(res.read().decode("utf-8"))

        if data.get("results"):
            result = data["results"][0]
            address = result["address1"] + result["address2"] + result["address3"]
            return f"{address}です。"
        return "その郵便番号は見つかりませんでした。"
    except Exception:
        return "検索に失敗しました。"


def get_f1_embed(driver_key):
    key = driver_key.lower()

    drivers_data = {
        # --- Red Bull Racing ---
        "max": {
            "name": "Max Verstappen", "team": "Red Bull Racing", "color": 0x0600EF,
            "prixs": "233", "points": "3444.5", "wins": "71", "champs": "4", "podiums": "127", "dnfs": "33",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png.transform/2col/image.png",
        },
        "tsunoda": {
            "name": "Yuki Tsunoda", "team": "Red Bull Racing", "color": 0x0600EF,
            "prixs": "111", "points": "124", "wins": "0", "champs": "0", "podiums": "0", "dnfs": "15",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/Y/YUKTSU01_Yuki_Tsunoda/yuktsu01.png.transform/2col/image.png",
        },
        # --- Mercedes ---
        "russell": {
            "name": "George Russell", "team": "Mercedes", "color": 0x00D2BE,
            "prixs": "152", "points": "1033", "wins": "5", "champs": "0", "podiums": "24", "dnfs": "19",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/G/GEORUS01_George_Russell/georus01.png.transform/2col/image.png",
        },
        "kimi": {
            "name": "Kimi Antonelli", "team": "Mercedes", "color": 0x00D2BE,
            "prixs": "24", "points": "150", "wins": "0", "champs": "0", "podiums": "3", "dnfs": "4",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/A/ANDANT01_Andrea_Kimi_Antonelli/andant01.png.transform/2col/image.png",
        },
        # --- Ferrari ---
        "leclerc": {
            "name": "Charles Leclerc", "team": "Ferrari", "color": 0xDC0000,
            "prixs": "171", "points": "1672", "wins": "8", "champs": "0", "podiums": "50", "dnfs": "23",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/C/CHALEC01_Charles_Leclerc/chalec01.png.transform/2col/image.png",
        },
        "lewis": {
            "name": "Lewis Hamilton", "team": "Ferrari", "color": 0xDC0000,
            "prixs": "380", "points": "5018.5", "wins": "105", "champs": "7", "podiums": "202", "dnfs": "34",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/L/LEWHAM01_Lewis_Hamilton/lewham01.png.transform/2col/image.png",
        },
        # --- McLaren ---
        "norris": {
            "name": "Lando Norris", "team": "McLaren", "color": 0xFF8700,
            "prixs": "152", "points": "1430", "wins": "11", "champs": "1", "podiums": "44", "dnfs": "13",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png.transform/2col/image.png",
        },
        "piastri": {
            "name": "Oscar Piastri", "team": "McLaren", "color": 0xFF8700,
            "prixs": "70", "points": "799", "wins": "9", "champs": "0", "podiums": "26", "dnfs": "4",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/O/OSCPIA01_Oscar_Piastri/oscpia01.png.transform/2col/image.png",
        },
        # --- Aston Martin ---
        "alonso": {
            "name": "Fernando Alonso", "team": "Aston Martin", "color": 0x006F62,
            "prixs": "427", "points": "2393", "wins": "32", "champs": "2", "podiums": "106", "dnfs": "83",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/F/FERALO01_Fernando_Alonso/feralo01.png.transform/2col/image.png",
        },
        "stroll": {
            "name": "Lance Stroll", "team": "Aston Martin", "color": 0x006F62,
            "prixs": "190", "points": "325", "wins": "0", "champs": "0", "podiums": "3", "dnfs": "31",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/L/LANSTR01_Lance_Stroll/lanstr01.png.transform/2col/image.png",
        },
        # --- Alpine ---
        "gasly": {
            "name": "Pierre Gasly", "team": "Alpine", "color": 0x0090FF,
            "prixs": "177", "points": "458", "wins": "1", "champs": "0", "podiums": "5", "dnfs": "26",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/P/PIEGAS01_Pierre_Gasly/piegas01.png.transform/2col/image.png",
        },
        "colapinto": {
            "name": "Franco Colapinto", "team": "Alpine", "color": 0x0090FF,
            "prixs": "27", "points": "5", "wins": "0", "champs": "0", "podiums": "0", "dnfs": "3",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/F/FRACOL01_Franco_Colapinto/fracol01.png.transform/2col/image.png",
        },
        # --- Williams ---
        "albon": {
            "name": "Alexander Albon", "team": "Williams", "color": 0x005AFF,
            "prixs": "128", "points": "313", "wins": "0", "champs": "0", "podiums": "2", "dnfs": "22",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/A/ALEALB01_Alexander_Albon/alealb01.png.transform/2col/image.png",
        },
        "sainz": {
            "name": "Carlos Sainz", "team": "Williams", "color": 0x005AFF,
            "prixs": "230", "points": "1336.5", "wins": "4", "champs": "0", "podiums": "29", "dnfs": "42",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/C/CARSAI01_Carlos_Sainz/carsai01.png.transform/2col/image.png",
        },
        # --- Racing Bulls (VCARB) ---
        "lawson": {
            "name": "Liam Lawson", "team": "Racing Bulls", "color": 0x6692FF,
            "prixs": "35", "points": "44", "wins": "0", "champs": "0", "podiums": "0", "dnfs": "6",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/L/LIALAW01_Liam_Lawson/lialaw01.png.transform/2col/image.png",
        },
        "hadjar": {
            "name": "Isack Hadjar", "team": "Racing Bulls", "color": 0x6692FF,
            "prixs": "23", "points": "51", "wins": "0", "champs": "0", "podiums": "1", "dnfs": "2",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/I/ISAHAD01_Isack_Hadjar/isahad01.png.transform/2col/image.png",
        },
        # --- Kick Sauber ---
        "hulkenberg": {
            "name": "Nico Hulkenberg", "team": "Kick Sauber", "color": 0x52E252,
            "prixs": "251", "points": "622", "wins": "0", "champs": "0", "podiums": "1", "dnfs": "44",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/N/NICHUL01_Nico_Hulkenberg/nichul01.png.transform/2col/image.png",
        },
        "bortoleto": {
            "name": "Gabriel Bortoleto", "team": "Kick Sauber", "color": 0x52E252,
            "prixs": "24", "points": "19", "wins": "0", "champs": "0", "podiums": "0", "dnfs": "5",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/G/GABBOR01_Gabriel_Bortoleto/gabbor01.png.transform/2col/image.png",
        },
        # --- Haas ---
        "ocon": {
            "name": "Esteban Ocon", "team": "Haas F1 Team", "color": 0xFFFFFF,
            "prixs": "180", "points": "483", "wins": "1", "champs": "0", "podiums": "4", "dnfs": "25",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/E/ESTOCO01_Esteban_Ocon/estoco01.png.transform/2col/image.png",
        },
        "bearman": {
            "name": "Oliver Bearman", "team": "Haas F1 Team", "color": 0xFFFFFF,
            "prixs": "27", "points": "48", "wins": "0", "champs": "0", "podiums": "0", "dnfs": "3",
            "img": "https://media.formula1.com/content/dam/fom-website/drivers/O/OLIBEA01_Oliver_Bearman/olibea01.png.transform/2col/image.png",
        },
    }

    if key in drivers_data:
        data = drivers_data[key]
        embed = discord.Embed(title=data["name"], description=data["team"], color=data["color"])
        embed.add_field(name="総出走数", value=data["prixs"], inline=True)
        embed.add_field(name="ポイント", value=data["points"], inline=True)
        embed.add_field(name="優勝回数", value=data["wins"], inline=True)
        if int(data["champs"]) > 0:
            embed.add_field(name="年間チャンピオン", value=data["champs"], inline=True)
        embed.add_field(name="表彰台", value=data["podiums"], inline=True)
        embed.add_field(name="DNFs", value=data["dnfs"], inline=True)
        embed.set_thumbnail(url=data["img"])
        return embed

    return "対象のドライバー名を入力してください。(ENDで終了)"


# --- Response Router ---
def generateResponse(inputText):
    global state

    # Active task-oriented state handling
    if state == State.JANKEN:
        if inputText == "END":
            state = State.DEFAULT
            return "じゃんけんを終了します。"
        return janken(inputText)

    if state == State.DAJARE:
        if inputText == "END":
            state = State.DEFAULT
            return "だじゃれモードを終了します。"
        return dajare(inputText)

    if state == State.ZIPCODE:
        if inputText == "END":
            state = State.DEFAULT
            return "ZIPCODEモードを終了します。"
        match = re.search(r"\d{7}", inputText)
        if match:
            return getAddressFromZip(match.group())
        return "7桁の郵便番号を入力してください。(ENDで終了)"

    if state == State.FORMULA1:
        if inputText == "END":
            state = State.DEFAULT
            return "FORMULA1モードを終了します。"
        return get_f1_embed(inputText)

    # Task-oriented triggers
    if "時間を教えて" in inputText:
        return datetime.datetime.now(JST).strftime("%Y/%m/%d %H:%M")

    elif "じゃんけん" in inputText:
        state = State.JANKEN
        return "じゃんけんをしましょう。グー，チョキ，パーから好きな手を選んで入力してください。(ENDで終了)"

    elif "だじゃれ" in inputText:
        state = State.DAJARE
        return "だじゃれモードに変更しました。(ENDで終了)"

    elif "郵便番号" in inputText:
        state = State.ZIPCODE
        return "郵便番号を入力してください。(ENDで終了)"

    elif "何が好き?" in inputText or "何が好き？" in inputText:
        icecream = ["チョコミント！", "クッキーアンドクリーム", "ストロベリーフレーバー"]
        return random.choice(icecream) + "よりもあなた！"

    elif "今日の運勢" in inputText or "占い" in inputText:
        kuji = ["大吉", "中吉", "小吉", "凶", "大凶"]
        return f"今日の運勢は... {random.choice(kuji)}です！"

    elif "疲れた" in inputText:
        relax_list = ["コーヒーを飲んで", "5分間寝て", "ストレッチをして", "散歩に行って"]
        return random.choice(relax_list) + "リラックスしましょう！"

    elif "F1" in inputText or "f1" in inputText:
        state = State.FORMULA1
        return (
            "ドライバーを入力してください。(ENDで終了)\n"
            "[Max]\n[Tsunoda]\n[Russell]\n[Kimi]\n[Leclerc]\n[Lewis]\n"
            "[Norris]\n[Piastri]\n[Alonso]\n[Stroll]\n[Gasly]\n[Colapinto]\n"
            "[Albon]\n[Sainz]\n[Lawson]\n[Hadjar]\n[Hulkenberg]\n[Bortoleto]\n"
            "[Ocon]\n[Bearman]"
        )

    # Non-task-oriented chat (scored candidate selection)
    candidateList.clear()
    wordlist = fugashiAnalyzer(inputText)
    generateResponseByRule(inputText)
    generateResponseByInputTopic(wordlist)
    generateOtherResponse()
    generateNegaposiResponse(inputText)

    ret = "デフォルト"
    maxScore = -1.0
    for cdd in candidateList:
        cdd.print()
        if cdd.score > maxScore:
            ret = cdd.response
            maxScore = cdd.score
    return ret


# --- Discord Client Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"ログインしました: {client.user}")


@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.channel.name != settings.channel_name:
        return

    text = message.content.strip()
    if not text:
        return

    try:
        output = generateResponse(text)
        if isinstance(output, discord.Embed):
            await message.channel.send(embed=output)
        else:
            await message.channel.send(f"{output}\nフラグ：{state}")
    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    setupKeywordMatchingRule()
    client.run(settings.token)