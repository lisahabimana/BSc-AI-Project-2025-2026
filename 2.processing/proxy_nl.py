import json
import re
import time
import ast
import openai
import os
from openai import RateLimitError
from json_repair import repair_json
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
import pandas as pd
from pypdf import PdfReader
from langchain_core.prompts import PromptTemplate

#RETRIEVE KEYS FROM ENVIRONMENT
gpt_main_key = os.getenv("GPT_MAIN_KEY")
print(gpt_main_key)
#LOAD MODELS
models = {
    "gpt52": ChatOpenAI( #gpt5
    model="gpt-5.2-pro",
    api_key=gpt_main_key,
    temperature=0
    )
}

#LOAD PROMPT
prompt = PromptTemplate(
    input_variables=["report"],
    template="""
Je bent een expert in het beoordelen van onderwijskundige documenten.

Je MOET het verslag beoordelen met de STRIKTE rubric hieronder.
Raad niet. Volg de definities exact.

RUBRIC

1. Bevat het OPP actuele en relevante informatie?
Score:
0 = Verouderde of irrelevante informatie.
1 = Grotendeels actueel.
2 = Volledig actueel en relevant.
Toelichting:
Neem recente toetsgegevens en recente observaties mee.
Vermijd verouderde schoolhistorie, tenzij die nog relevant is.
Rode vlaggen: oude groepsbeschrijvingen, verlopen IQ-onderzoek zonder toelichting, lange historische beschrijvingen.

2. Wordt gedrag observeerbaar en specifiek beschreven?
Score:
0 = Gedrag is abstract of labelend beschreven.
1 = Gedrag is deels concreet.
2 = Gedrag is observeerbaar, specifiek en meetbaar.
Toelichting:
Beschrijf wat de leerling daadwerkelijk doet.
Goed voorbeeld: "start niet zelfstandig aan een taak zonder visuele instructie."
Slechte voorbeelden: "is ongemotiveerd." "heeft gedragsproblemen."

3. Leidt het OPP tot concrete onderwijskundige acties?
Score:
0 = Er zijn geen concrete onderwijskundige acties beschreven.
1 = De onderwijskundige acties zijn slechts deels beschreven.
2 = Het OPP biedt duidelijke, direct toepasbare acties voor de leerkracht.
Toelichting:
Vraag jezelf af: wat moet de leerkracht morgen anders doen?
Aanbevelingen moeten direct bruikbaar zijn in de praktijk.

4. Is er een logische verbinding tussen observaties, analyse, ondersteuningsbehoeften en uitstroomperspectief?
Score:
0 = Onderdelen staan los van elkaar.
1 = Beperkte logische samenhang.
2 = Sterke en consistente logische samenhang door het hele OPP.
Toelichting:
Observaties moeten de vastgestelde behoeften verklaren.
Ondersteuningsbehoeften moeten het voorgestelde uitstroomperspectief onderbouwen.

5. Is de taal professioneel, duidelijk en goed leesbaar?
Score:
0 = Vage taal of veel jargon.
1 = Overwegend begrijpelijk.
2 = Duidelijk, compact en professioneel.
Toelichting:
Vermijd oordelende taal zoals "lui," "zwak" of "kan niet."
Kies liever voor behoeftegerichte formuleringen zoals "heeft ondersteuning nodig bij…" of "laat zien dat…"

6. Wordt het ontwikkelingsfunctioneren passend beschreven?
Score:
0 = Algemeen of beschrijvend.
1 = Deels concreet.
2 = Observeerbaar, contextgebonden en onderwijskundig relevant.
Toelichting:
Denk aan:
sociaal-emotioneel functioneren;
executieve functies;
communicatie;
taakgedrag;
zelfstandigheid.
Goed voorbeeld: "raakt overprikkeld tijdens groepswisselingen."

7. Wordt het didactisch functioneren betekenisvol beschreven?
Score:
0 = Alleen scores.
1 = Scores met beperkte interpretatie.
2 = Scores met interpretatie en betekenis voor het onderwijs.
Toelichting:
Leg uit wat de leerling in de klas kan, niet alleen welke testscore is behaald.

8. Zijn bevorderende factoren concreet en onderwijsrelevant?
Score:
0 = Algemeen of irrelevante sterktes.
1 = Redelijk concreet.
2 = Observeerbaar, functioneel en relevant voor het leren.
Toelichting:
Beschrijf sterktes die het leren ondersteunen.
Goed: "profiteert van visuele ondersteuning."
Goed: "werkt taakgerichter bij voorspelbare structuur."
Vermijd persoonlijkheidskenmerken zoals "is lief" of "heeft humor."

9. Zijn belemmerende factoren concreet en onderwijsrelevant?
Score:
0 = Algemene beschrijvingen of alleen diagnostische labels.
1 = Gedeeltelijk concreet.
2 = Observeerbaar, functioneel en onderwijsrelevant.
Toelichting:
Beschrijf zichtbare belemmeringen in plaats van diagnoses.
Goed: "verliest overzicht bij opdrachten met meerdere stappen."
Goed: "raakt afgeleid door auditieve prikkels."
Vermijd: ADHD, autisme, gedragsproblemen.

10. Zijn de onderwijs- en ondersteuningsbehoeften concreet en uitvoerbaar?
Score:
0 = Problemen worden beschreven in plaats van ondersteuningsbehoeften.
1 = De behoeften zijn deels gespecificeerd.
2 = De behoeften zijn concreet, leerling-specifiek en direct uitvoerbaar.
Toelichting:
Ondersteuningsbehoeften moeten:
Als behoefte geformuleerd zijn.
Beschrijven wie iets doet, wat er gebeurt, waar, wanneer en hoe.
Leerling-specifiek zijn.
Verder gaan dan basisondersteuning, tenzij dat expliciet onderbouwd is.
Direct handelbaar zijn voor professionals.

11. Is de intensiteit van de ondersteuning duidelijk beschreven?
Score:
0 = Geen duidelijkheid over intensiteit.
1 = Globale aanduiding.
2 = Duidelijk onderbouwde frequentie en intensiteit.
Toelichting:
Geef aan wanneer en hoe vaak ondersteuning nodig is.
Voorbeelden:
Dagelijks.
Tijdens overgangsmomenten.
Voortdurende begeleiding bij taakstart.

12. Geeft het integratief beeld daadwerkelijk samenhang?
Score:
0 = Opsomming of herhaling.
1 = Enige analyse.
2 = Sterke integratieve analyse.
Toelichting:
Een sterk integratief beeld:
Verbindt meerdere ontwikkelingsdomeinen.
Verklaart oorzaak-gevolgrelaties.
Verbindt observaties aan ondersteuning.
Beschrijft onderwijsimplicaties.
Vermijdt herhaling van losse onderdelen of lange narratieve beschrijvingen.

13. Is het uitstroomperspectief logisch onderbouwd?
Score:
0 = Geen onderbouwing.
1 = Beperkte onderbouwing.
2 = Logisch en consistent onderbouwd.
Toelichting:
Kijk naar de relatie met didactisch functioneren, executieve functies, zelfstandigheid en ondersteuningsbehoeften.

14. Is het document intern consistent?
Score:
0 = Tegenstrijdig.
1 = Kleine inconsistenties.
2 = Volledig consistent.
Toelichting:
Controleer of de onderdelen elkaar ondersteunen.
Voorbeelden van inconsistentie:
hoge zelfstandigheid tegenover voortdurende begeleiding;
uitstroom naar een hoger niveau tegenover zeer lage didactische gegevens.

15. Vermijdt het OPP generieke taal?
Score:
0 = Veel generieke formuleringen.
1 = Enkele generieke formuleringen.
2 = Vrijwel volledig specifiek.
Toelichting:
Vervang vage formuleringen door concrete beschrijvingen.
Vermijd:
"heeft structuur nodig."
"heeft behoefte aan duidelijkheid."
"positieve feedback."

16. Is de taal primair behoeftegericht en niet probleemgericht?
Score:
0 = Veel probleemgerichte taal.
1 = Enigszins probleemgericht.
2 = Overwegend behoeftegericht.
Toelichting:
Beschrijf observeerbaar gedrag en onderwijsbehoeften in plaats van tekortkomingen of diagnoses te benadrukken.

17. Is het OPP compact en gefocust?
Score:
0 = Te lang of te weinig gericht.
1 = Redelijk compact.
2 = Compact en gefocust op relevante informatie.
Toelichting:
Neem alleen informatie op die bijdraagt aan het begrijpen van de onderwijsbehoeften en ondersteuning van de leerling. Vermijd onnodige herhaling of uitgebreide historische achtergrond.

TAAK

Verslag:
{report}

Geef ALLEEN geldige JSON terug in exact dit formaat:

{{
  "criteria": [17 integers between 0-2],
  "total_score": integer,
  "feedback": "string"
}}

Regels:
- Gebruik alleen dubbele aanhalingstekens.
- Gebruik geen enkele aanhalingstekens.
- Plaats de JSON niet in markdown.
- Voeg geen extra tekst toe vóór of na de JSON.
"""
)

#EXTRACT TEXT FROM PDF
def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n\n".join(text).strip()

#DEBUG AND CLEAN FUNCTION FOR JSON
def clean_json_response(content):
    content = str(content).strip()
    content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = re.sub(r"^Here is the JSON:\s*", "", content, flags=re.IGNORECASE)
    return content.strip()


def extract_json_block(text):
    text = str(text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start:end + 1]


def parse_model_output(content):
    content = clean_json_response(content)
    content = extract_json_block(content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = json.loads(repair_json(content))
    if isinstance(data, list):
        dict_items = [item for item in data if isinstance(item, dict)]
        if not dict_items:
            return None, content
        data = dict_items[-1]
    if not isinstance(data, dict):
        return None, content
    return data, content

#LOAD FILES
pdf_dir = "data/reports/OPP"
pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))

#OUTPUT DIRECRTORY
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

#LOOP THROUGH MODEL & REPORTS THEN SAVE EVALUATIONS OF MODEL (NL)
for model_name, model in models.items():
    results = []
    out_path = OUT_DIR / f"{model_name}_results_nl.csv"

    for idx, pdf_path in enumerate(pdf_files[:24], start=1):
        report = extract_pdf_text(pdf_path)

        if not report:
            print(f"Empty report: {pdf_path.name}")
            continue

        try:
            response = model.invoke(prompt.format(report=report))
            content = response.content
            if isinstance(content, list):
                content = "\n".join(
                    item["text"] if isinstance(item, dict) and "text" in item else str(item)
                    for item in content
                )
            data, raw_output = parse_model_output(content)
        except RateLimitError as e:
            print(f"API quota stopped run at {pdf_path.name}: {e}")
            break
        except Exception as e:
            print(f"Unexpected error at {pdf_path.name}: {e}")
            continue

        if data is None:
            print(f"Parse failed: {pdf_path.name}")
            continue

        if not all(k in data for k in ("criteria", "total_score", "feedback")):
            print(f"Bad schema: {pdf_path.name}")
            print(f"Returned keys: {sorted(data.keys())}")
            continue

        if not isinstance(data["criteria"], list) or len(data["criteria"]) != 17:
            print(f"Bad criteria length: {pdf_path.name}")
            continue

        results.append({
            "report_number": idx,
            "pdf_name": pdf_path.name,
            "report": report,
            **{f"c{i+1}": data["criteria"][i] for i in range(17)},
            "total_score": data["total_score"],
            "feedback": data["feedback"],
            "raw_output": raw_output,
        })

        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"{model_name}: processed {idx}/{min(25, len(pdf_files))} -> {pdf_path.name}")

    print(f"{model_name}: collected {len(results)} valid rows")
    if results:
        pd.DataFrame(results).to_csv(out_path, index=False)
    else:
        print(f"No results to save for {model_name}")


print("Done")