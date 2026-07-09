"""
[1단계] DART 전체 법인 목록(상장 + 비상장 모두 포함)을 받아서
        corp_code 마스터 테이블을 만드는 코드.

실행하면 -> dart_corp_code_all.csv 파일이 생성됩니다.
"""

import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd

# ---- 본인 DART API 키 ----
API_KEY = "여기에_본인_DART_API_키를_입력하세요"

url = "https://opendart.fss.or.kr/api/corpCode.xml"
params = {"crtfc_key": API_KEY}
response = requests.get(url, params=params)

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    xml_data = z.read(z.namelist()[0])

root = ET.fromstring(xml_data)

corp_list = []
for corp in root.findall("list"):
    corp_list.append({
        "corp_code": corp.findtext("corp_code"),
        "corp_name": corp.findtext("corp_name"),
        "stock_code": corp.findtext("stock_code"),
        "modify_date": corp.findtext("modify_date"),
    })

df_all = pd.DataFrame(corp_list)
print(f"전체 법인 수(상장+비상장): {len(df_all)}")

df_all.to_csv("dart_corp_code_all.csv", index=False, encoding="utf-8-sig")
print("저장 완료 -> dart_corp_code_all.csv")
