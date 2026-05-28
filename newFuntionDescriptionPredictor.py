import pandas as pd
import json
import time
import random
from typing import List, Callable, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type
from Settings import get_uploaded_data_df
from langchain_openai import ChatOpenAI
import sys
import streamlit as st
from Settings import get_special_requirements
from langchain_google_genai import ChatGoogleGenerativeAI
from Settings import get_openai_key, get_google_api_key, get_selected_llm
DEBUG=False
class newFuntionDescriptionPredictor:

    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini", batch_size: int = 5):
        
        provider = get_selected_llm()
        # self.llm = ChatOpenAI(openai_api_key=openai_api_key, model=model_name)
        self.batch_size = batch_size
        if provider == "openai":
            self.llm = ChatOpenAI(
                openai_api_key=get_openai_key(),
                model=model_name,
               
            )
        elif provider == "google":
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=get_google_api_key(),
                model="gemini-2.5-flash", 
               
            )

    def build_prompt(self, batch: List[dict]) -> List:
        df = get_uploaded_data_df()
        user_reqs=get_special_requirements()
        user_req_promopt=""
        if len(user_reqs)>2:
           user_req_promopt = f"Please consider the user's special requirements to finalize the output. Give priority to these requirements:\n\n{user_reqs if user_reqs else 'No specific requirements provided.'}"

          #Group input data by MFH_Mapping
        grouped_data = {}
        for _, row in df.iterrows():
            mfh = str(row["MFH_Mapping"]).strip()
            legacy = str(row["LegacyFunctionPath"]).strip()
            descr = str(row["NewFunctionDescription"]).strip()

            if mfh and legacy and descr:
                grouped_data.setdefault(mfh, []).append({
                    "LegacyFunctionPath": legacy,
                    "MFH_Mapping": mfh,
                    "NewFunctionDescription": descr
                })

        prompt_examples = json.dumps(grouped_data, indent=2)

        system_prompt = SystemMessage(content=f"""
You are a maritime functional hierarchy expert.

Your task is to generate a clean, standardized `newFunctionDescription` for each component using strict formatting and extraction rules. This task requires precision, rule compliance, and consistency — not creativity.

---

🧩 Input Fields:
- `FuncDescr`: Raw name or label of the component.
- `LegacyFunctionPath`: A semicolon-separated functional hierarchy path.
- `MFHMapping`: Internal reference that defines the intended structure and component type.

---

⚙️ Rule-Based Generation Logic:

1. **Field Priority:**
   - Extract numeric and position identifiers from `LegacyFunctionPath` first.
   - Use `FuncDescr` only if `LegacyFunctionPath` provides no valid identifier.
   - Use `MFHMapping` as a structural template — follow its sequence of words and component types, but do **not blindly copy unresolved placeholders** like `NO XX`.

2. **Handling Placeholders in MFHMapping:**
   - MFHMapping may contain placeholders like `NO XX`, `(XX)`, `- XX`, or `YY`.
     - These placeholders must be replaced **only** if a valid identifier (e.g., `01`, `PS`, `SB`) is extracted from `LegacyFunctionPath` or `FuncDescr`.
     - If no valid identifier is found, **remove the entire placeholder segment** from the output.
     - Do not insert filler terms like `UNDETERMINED`, `UNKNOWN`, or leave `XX` in place.
     - Example:
       - Input MFHMapping: `"EMERGENCY GENERATOR NO XX"`
       - LegacyFunctionPath contains: `"GENERATOR 01"` → ✅ Output: `"EMERGENCY GENERATOR NO 01"`
       - No identifier found → ✅ Output: `"EMERGENCY GENERATOR"`

3. **Identifier Normalization:**
   Normalize extracted indicators using these rules:
   - `port`, `ps`, `pt` → `PS`
   - `starboard`, `sb`, `stbd`, `st` → `SB`
   - `fwd`, `forward` → `FWD`
   - `aft` → `AFT`
   - `center`, `cent`, `cnt` → `CENTER`
   Also extract numbers like: `01`, `02`, `NO 03`, etc.

---

 🔁 Special Note on Placeholder Resolution:

- When replacing placeholders in `MFHMapping`:
  - If the extracted identifier is a **number** (e.g., `01`, `02`), insert it with the "NO" prefix if present in the MFHMapping (e.g., `NO 01`).
  - If the identifier is a **position indicator** (e.g., `PS`, `SB`, `FWD`), **do not prefix it with "NO"**. Insert it directly in its position within the structure.
  - Example corrections:
    - ✅ `NO 01` → valid
    - ❌ `NO SB` → invalid → ✅ should be `SB`

🚨 Fallback Use of MFHMapping Without Identifier:

- If `MFHMapping` contains a placeholder like `NO XX`, and no number or position can be extracted:
  - Remove only the placeholder segment (e.g., remove `NO XX`)
  - Retain the valid parts of the MFHMapping (e.g., `EMERGENCY GENERATOR`)
- Do **not** return `"UNDETERMINED"` unless:
  - The `MFHMapping` itself is invalid or empty
  - Or the input provides no meaningful component type at all


❗ Do not prefix position indicators like `PS`, `SB`, `FWD` with "NO".
- If `MFHMapping` contains `NO XX`, and only a position (not a number) is found, remove the `NO` and insert only the position.
- ❌ Invalid: `NO PS`, `NO SB`
- ✅ Valid: `PS`, `SB`


🔍 Positional Suffix Normalization:

- When scanning `FuncDescr` or `LegacyFunctionPath`, recognize suffix patterns like `#1PS`, `#2SB`, `GENERATOR 1 SB`, etc.
- Extract the **position indicator** (`PS`, `SB`, `FWD`, etc.) even if it is attached to a number or a special character.
- Normalize extracted positions as follows:
  - `port`, `ps`, `pt` → `PS`
  - `starboard`, `sb`, `stbd`, `st` → `SB`
  - `fwd`, `forward` → `FWD`
  - `aft` → `AFT`
  - `center`, `cent`, `cnt` → `CENTER`
- Example extractions:
  - `EMCY GENERATOR ENGINE #1PS` → `PS`
  - `#2SB` → `SB`
  - `GENERATOR NO.02 STBD` → `SB`


❗ Never prefix position indicators with `NO`. If the placeholder `NO XX` is in `MFHMapping` and only a position like `PS` or `SB` is found:
- Remove the entire `NO XX` segment.
- Insert the position indicator in the proper sequence without `NO`.

✅ Valid: `EMERGENCY GENERATOR PS`  
❌ Invalid: `EMERGENCY GENERATOR NO PS`


🔍 Position and Number Extraction (Advanced):

- Extract position indicators and numbers from `LegacyFunctionPath` and `FuncDescr`. Normalize as:

  - `port`, `ps`, `pt` → `PS`
  - `starboard`, `sb`, `stbd`, `st` → `SB`
  - `fwd`, `forward` → `FWD`
  - `aft` → `AFT`
  - `center`, `cent`, `cnt` → `CENTER`

- Recognize **embedded position indicators or numbers** in patterns like:
  - `#1PS` → `PS`
  - `#2SB` → `SB`
  - `ENGINE NO.02` → `NO 02`

- If only a position (e.g., `PS`, `SB`) is found and `MFHMapping` includes `NO XX`, **remove `NO`** and insert just the position:
  - ✅ `EMERGENCY GENERATOR PS`
  - ❌ `EMERGENCY GENERATOR NO PS`

- Never use position indicators unless they are clearly extractable from the input.
- If nothing valid is found (neither position nor number), drop the placeholder and return the base MFHMapping (e.g., `EMERGENCY GENERATOR`).


🎯 Output Formatting Rules:

- The final output must be in **UPPER CASE**.
- Follow the structure and order of `MFHMapping`, but omit unresolved placeholder segments.
- Do not include:
  - Placeholder text: `XX`, `YY`, `UNDETERMINED`, etc.
  - Vendor or manufacturer names (unless part of MFHMapping).
- Return one `newFunctionDescription` per input.
- Keep output concise and technically correct.
- Return exactly one output record for each input record, preserving the original input order.if input records are 10 then output should contain 10 records.
- Do not skip, omit, or combine rows.
- The number of output objects **must match the number of input objects exactly** — in order and count.

---

📘 Examples (Follow Structure):

Use the following JSON examples for formatting and logic reference:
{prompt_examples}

---

🔒 Hard Constraints:

- Return the same number of records as input.
- Preserve the input order.
- Return **only one output** per input.
- If no meaningful description can be created, return:
  {{ "newFunctionDescription": "UNDETERMINED" }}

---
{user_req_promopt}

📤 Output Format (Strict):

Return a pure JSON array only:
[
   {{ "index": 0, "newFunctionDescription": "..." }},
  ...
]
""")


       

        human_prompt = HumanMessage(content=f"Here is the input data to process:please follow all rules..\n\n{json.dumps(batch, indent=2)}")
       
        if DEBUG:
            st.subheader("🔍 Prompt Preview")
            #st.code(system_prompt.content, language="markdown")
            st.code(human_prompt.content, language="markdown")
        return [system_prompt, human_prompt]


    @retry(
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(Exception)
    )
    def predict_batch(self, batch_df: pd.DataFrame) -> List[dict]:
        #input_records = batch_df[["FuncDescr", "legacyfunctionidpath", "MFHMapping"]].fillna("").to_dict(orient="records")
        input_records = batch_df[["FuncDescr", "legacyfunctionidpath", "MFHMapping"]].fillna("").reset_index(drop=True).to_dict(orient="records")
        for i, r in enumerate(input_records):
            r["index"] = i

        messages = self.build_prompt(input_records)

        try:
            response = self.llm(messages).content
            if DEBUG:
                st.code(response, language="markdown")

            cleaned_response = response.strip().strip("`").strip()
            if cleaned_response.startswith("json"):
                cleaned_response = cleaned_response[4:].strip()

            predictions = json.loads(cleaned_response)

            # Sort back by index if available
            if isinstance(predictions[0], dict) and "index" in predictions[0]:
                predictions.sort(key=lambda x: x["index"])
                for p in predictions:
                    p.pop("index", None)

            # 🛡 Force size match
            if len(predictions) < len(input_records):
                missing = len(input_records) - len(predictions)
                predictions += [{"newFunctionDescription": "UNDETERMINED"} for _ in range(missing)]
            elif len(predictions) > len(input_records):
                predictions = predictions[:len(input_records)]

            return predictions

        except Exception as e:
            print(f"❌ LLM Error: {e}", file=sys.stderr, flush=True)
            raise e

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        streamlit_callback: Optional[Callable[[int, int], None]] = None
    ) -> pd.DataFrame:
        df = df.copy()
        df["__key__"] = df[["FuncDescr", "legacyfunctionidpath", "MFHMapping"]].fillna("").astype(str).agg("||".join, axis=1)

        unique_df = df.drop_duplicates(subset="__key__")
        unique_keys = unique_df["__key__"].tolist()
        unique_records = unique_df[["FuncDescr", "legacyfunctionidpath", "MFHMapping"]].copy()

        predictions = []
        total = len(unique_records)
        for i in range(0, total, self.batch_size):
            batch_df = unique_records.iloc[i:i + self.batch_size]
            try:
                batch_predictions = self.predict_batch(batch_df)
            except Exception:
                batch_predictions = [{"newFunctionDescription": "UNKNOWN"} for _ in range(len(batch_df))]

            predictions.extend(batch_predictions)
            if streamlit_callback:
                streamlit_callback(min(i + self.batch_size, total), total)

            time.sleep(random.uniform(0.5, 1.5))

        unique_predictions = [p.get("newFunctionDescription", "UNKNOWN") for p in predictions]
        key_to_prediction = dict(zip(unique_keys, unique_predictions))

        # Defensive check: handle missing keys
        df["newFunctionDescription"] = df["__key__"].apply(lambda k: key_to_prediction.get(k, "UNDETERMINED"))

        df.drop(columns=["__key__"], inplace=True)

        return df