#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic EHR Dataset Generator
--------------------------------
Genera datos clínicos sintéticos y coherentes para simular un episodio de hospitalización
y permitir entrenar/probar un agente de resúmenes evolutivos al alta.
Salida por defecto: CSVs en el directorio "output". Opcionalmente, una base SQLite.

Tablas generadas (CSV/SQLite):
- patients
- admissions
- evolutions
- vitals
- labs
- meds
- procedures
- notes
- discharges

Uso:
    python synthetic_ehr_generator.py --patients 200 --seed 42 --format both
"""

import argparse
import os
import random
import string
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import sqlite3

# -------------------------- Utilidades generales --------------------------

SEX_CHOICES = ["M", "F"]
SERVICES = [
    "Medicina Interna",
    "Neumología",
    "Cardiología",
    "Neurología",
    "Traumatología",
    "Cirugía General",
    "Oncología",
    "Pediatría",
    "UCI",
    "Geriatría",
]

MOTIVOS_INGRESO = [
    "Neumonía adquirida en la comunidad",
    "Insuficiencia cardíaca descompensada",
    "Accidente cerebrovascular isquémico",
    "Fractura de cadera",
    "Dolor abdominal / Apendicitis",
    "COVID-19",
    "Crisis asmática",
    "Cetoacidosis diabética",
    "Sepsis de origen urinario",
    "Exacerbación EPOC",
]

ALERGIAS = ["Ninguna conocida", "Penicilina", "AINEs", "Mariscos", "Latex", "Sulfas"]

ANTECEDENTES = [
    "HTA",
    "DM2",
    "EPOC",
    "FA",
    "IRC",
    "Cardiopatía isquémica",
    "Obesidad",
    "Tabaquismo",
    "Sin antecedentes relevantes",
]

PROCEDIMIENTOS = [
    "Catéter venoso periférico",
    "Catéter venoso central",
    "Toracocentesis",
    "Paracentesis",
    "Intubación orotraqueal",
    "Broncoscopia",
    "Laparoscopia diagnóstica",
    "Cirugía abierta",
    "Drenaje pleural",
    "Colocación sonda nasogástrica",
]

FARMACOS = [
    "Ceftriaxona",
    "Amoxicilina/Ácido clavulánico",
    "Azitromicina",
    "Piperacilina/Tazobactam",
    "Levofloxacino",
    "Heparina de bajo peso molecular",
    "Paracetamol",
    "Dexametasona",
    "Omeprazol",
    "Furosemida",
    "Insulina rápida",
    "Insulina basal",
    "Salbutamol",
    "Metronidazol",
]

VIAS = ["Oral", "IV", "IM", "SC", "Inhalada"]

ESTADO_CLINICO = ["Crítico", "Empeoramiento", "Estable", "Mejoría"]

LAB_TYPES = [
    "Hemograma",
    "Bioquímica",
    "Gasometría",
    "PCR",
    "Procalcitonina",
    "Radiografía de tórax",
    "TAC torácico",
]


# Generador simple de ID legible
def uid(prefix: str, n: int = 8) -> str:
    return f"{prefix}_" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=n)
    )


# Fechas aleatorias con coherencia
def rand_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    sec = random.randrange(int(delta.total_seconds()) + 1)
    return start + timedelta(seconds=sec)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# -------------------------- Generadores de tablas --------------------------


def generate_patients(n_patients: int) -> pd.DataFrame:
    rows = []
    today = datetime.today()
    for _ in range(n_patients):
        pid = uid("P")
        age = int(clamp(np.random.normal(65, 18), 0, 100))  # distribución aproximada
        sex = random.choice(SEX_CHOICES)
        # Antecedentes: 0-3 aleatorios
        antecedentes = ", ".join(
            sorted(random.sample(ANTECEDENTES, k=random.randint(1, 3)))
        )
        alergias = random.choice(ALERGIAS)
        rows.append(
            {
                "patient_id": pid,
                "edad": age,
                "sexo": sex,
                "antecedentes": antecedentes,
                "alergias": alergias,
            }
        )
    return pd.DataFrame(rows)


def generate_admissions(
    patients_df: pd.DataFrame, mean_stay_days: float = 7.0
) -> pd.DataFrame:
    rows = []
    base_start = datetime(2024, 1, 1)
    base_end = datetime(2025, 10, 1)
    for _, p in patients_df.iterrows():
        # 1-2 ingresos por paciente
        n_adm = 1 if random.random() < 0.8 else 2
        for _ in range(n_adm):
            aid = uid("A")
            fecha_ingreso = rand_date(base_start, base_end)
            # estancia ~ N(mean_stay_days, 3)
            stay = max(1, int(np.random.normal(mean_stay_days, 3)))
            fecha_alta = fecha_ingreso + timedelta(days=stay)
            rows.append(
                {
                    "admission_id": aid,
                    "patient_id": p["patient_id"],
                    "fecha_ingreso": fecha_ingreso,
                    "fecha_alta": fecha_alta,
                    "motivo_ingreso": random.choice(MOTIVOS_INGRESO),
                    "servicio": random.choice(SERVICES),
                }
            )
    df = (
        pd.DataFrame(rows)
        .sort_values(by=["patient_id", "fecha_ingreso"])
        .reset_index(drop=True)
    )
    return df


def generate_evolutions(admissions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, a in admissions_df.iterrows():
        start = (
            a["fecha_ingreso"].to_pydatetime()
            if isinstance(a["fecha_ingreso"], pd.Timestamp)
            else a["fecha_ingreso"]
        )
        end = (
            a["fecha_alta"].to_pydatetime()
            if isinstance(a["fecha_alta"], pd.Timestamp)
            else a["fecha_alta"]
        )
        n_days = (end.date() - start.date()).days + 1
        # 1-2 evoluciones por día
        evo_dates = []
        for d in range(n_days):
            day = start.date() + timedelta(days=d)
            n_evo = 1 if random.random() < 0.7 else 2
            for _ in range(n_evo):
                # hora aleatoria ese día
                dt = datetime.combine(day, datetime.min.time()) + timedelta(
                    hours=random.randint(7, 22), minutes=random.randint(0, 59)
                )
                evo_dates.append(dt)
        evo_dates.sort()
        # Estado clínico con tendencia a mejorar hacia el alta
        for i, dt in enumerate(evo_dates):
            prog = i / max(1, len(evo_dates) - 1)
            # mayor probabilidad de "Mejoría" cerca del alta
            state_weights = np.array(
                [
                    0.05 + 0.05 * (1 - prog),
                    0.2 + 0.1 * (1 - prog),
                    0.5 - 0.2 * (prog),
                    0.25 + 0.35 * (prog),
                ]
            )
            state_weights = state_weights / state_weights.sum()
            estado = random.choices(ESTADO_CLINICO, weights=state_weights, k=1)[0]
            rows.append(
                {
                    "evolution_id": uid("E", 10),
                    "admission_id": a["admission_id"],
                    "fecha": dt,
                    "estado_clinico": estado,
                    "descripcion": f"Evolución {estado.lower()}. Control de síntomas y tolerancia a tratamiento.",
                }
            )
    return pd.DataFrame(rows).sort_values("fecha").reset_index(drop=True)


def generate_vitals(admissions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, a in admissions_df.iterrows():
        start = (
            a["fecha_ingreso"]
            if isinstance(a["fecha_ingreso"], datetime)
            else a["fecha_ingreso"].to_pydatetime()
        )
        end = (
            a["fecha_alta"]
            if isinstance(a["fecha_alta"], datetime)
            else a["fecha_alta"].to_pydatetime()
        )
        # 2-4 registros por día
        n_days = (end.date() - start.date()).days + 1
        for d in range(n_days):
            for _ in range(random.randint(2, 4)):
                dt = datetime.combine(
                    start.date() + timedelta(days=d), datetime.min.time()
                ) + timedelta(
                    hours=random.randint(6, 22), minutes=random.randint(0, 59)
                )
                temp = round(np.random.normal(37.0, 0.6), 1)
                fc = int(clamp(np.random.normal(85, 15), 40, 160))
                sist = int(clamp(np.random.normal(120, 15), 80, 200))
                diast = int(clamp(np.random.normal(75, 10), 50, 120))
                spo2 = int(clamp(np.random.normal(95, 3), 70, 100))
                rr = int(clamp(np.random.normal(18, 4), 8, 40))
                rows.append(
                    {
                        "vital_id": uid("V"),
                        "admission_id": a["admission_id"],
                        "fecha": dt,
                        "temperatura_c": temp,
                        "frecuencia_cardiaca_lpm": fc,
                        "presion_arterial": f"{sist}/{diast}",
                        "saturacion_o2_pct": spo2,
                        "respiraciones_rpm": rr,
                    }
                )
    return pd.DataFrame(rows).sort_values("fecha").reset_index(drop=True)


def generate_labs(admissions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, a in admissions_df.iterrows():
        start = (
            a["fecha_ingreso"]
            if isinstance(a["fecha_ingreso"], datetime)
            else a["fecha_ingreso"].to_pydatetime()
        )
        end = (
            a["fecha_alta"]
            if isinstance(a["fecha_alta"], datetime)
            else a["fecha_alta"].to_pydatetime()
        )
        n = random.randint(3, 8)
        for _ in range(n):
            dt = rand_date(start, end)
            lab_type = random.choice(LAB_TYPES)
            # Crear valores sintéticos simples
            if lab_type == "Hemograma":
                hemoglobina = round(np.random.normal(13.5, 1.8), 1)
                leucocitos = int(clamp(np.random.normal(9_000, 3_000), 2_000, 30_000))
                plaquetas = int(
                    clamp(np.random.normal(250_000, 60_000), 50_000, 700_000)
                )
                resultados = (
                    f"Hb: {hemoglobina} g/dL; Leu: {leucocitos}/µL; Plq: {plaquetas}/µL"
                )
            elif lab_type == "Bioquímica":
                sodio = int(clamp(np.random.normal(138, 4), 120, 160))
                potasio = round(np.random.normal(4.2, 0.5), 1)
                creatinina = round(np.random.normal(1.0, 0.3), 2)
                resultados = (
                    f"Na: {sodio} mmol/L; K: {potasio} mmol/L; Cr: {creatinina} mg/dL"
                )
            elif lab_type == "Gasometría":
                ph = round(np.random.normal(7.40, 0.05), 2)
                pco2 = int(clamp(np.random.normal(40, 6), 20, 80))
                po2 = int(clamp(np.random.normal(75, 12), 40, 120))
                resultados = f"pH: {ph}; pCO2: {pco2} mmHg; pO2: {po2} mmHg"
            elif lab_type == "PCR":
                pcr = round(abs(np.random.normal(30, 25)), 1)
                resultados = f"PCR: {pcr} mg/L"
            elif lab_type == "Procalcitonina":
                pct = round(abs(np.random.normal(0.6, 0.8)), 2)
                resultados = f"PCT: {pct} ng/mL"
            else:
                resultados = "Hallazgos compatibles con proceso infeccioso en lóbulo inferior derecho."
            rows.append(
                {
                    "result_id": uid("R"),
                    "admission_id": a["admission_id"],
                    "fecha": dt,
                    "tipo_examen": lab_type,
                    "resultados": resultados,
                    "interpretacion": "Resultados valorados en contexto clínico.",
                }
            )
    return pd.DataFrame(rows).sort_values("fecha").reset_index(drop=True)


def generate_meds(admissions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, a in admissions_df.iterrows():
        start = (
            a["fecha_ingreso"]
            if isinstance(a["fecha_ingreso"], datetime)
            else a["fecha_ingreso"].to_pydatetime()
        )
        end = (
            a["fecha_alta"]
            if isinstance(a["fecha_alta"], datetime)
            else a["fecha_alta"].to_pydatetime()
        )
        base_n = random.randint(2, 5)
        for _ in range(base_n):
            f = random.choice(FARMACOS)
            via = random.choice(VIAS)
            t_start = rand_date(
                start, end - timedelta(days=0)
            )  # podría finalizar el mismo día
            t_end = t_start + timedelta(days=max(0, int(np.random.exponential(2))))
            if t_end > end:
                t_end = end
            dosis = random.choice(
                ["500 mg/8h", "1 g/12h", "2 g/24h", "40 mg/24h", "10 UI según pauta"]
            )
            rows.append(
                {
                    "treatment_id": uid("T"),
                    "admission_id": a["admission_id"],
                    "farmaco": f,
                    "dosis": dosis,
                    "via_administracion": via,
                    "fecha_inicio": t_start,
                    "fecha_fin": t_end,
                    "motivo_cambio": ""
                    if random.random() < 0.6
                    else "Ajuste por respuesta clínica",
                }
            )
    return pd.DataFrame(rows).sort_values("fecha_inicio").reset_index(drop=True)


def generate_procedures(admissions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, a in admissions_df.iterrows():
        start = (
            a["fecha_ingreso"]
            if isinstance(a["fecha_ingreso"], datetime)
            else a["fecha_ingreso"].to_pydatetime()
        )
        end = (
            a["fecha_alta"]
            if isinstance(a["fecha_alta"], datetime)
            else a["fecha_alta"].to_pydatetime()
        )
        if random.random() < 0.7:
            n = random.randint(1, 3)
            for _ in range(n):
                dt = rand_date(start, end)
                rows.append(
                    {
                        "procedure_id": uid("Prc"),
                        "admission_id": a["admission_id"],
                        "fecha": dt,
                        "procedimiento": random.choice(PROCEDIMIENTOS),
                        "descripcion": "Realizado sin incidencias. Control posterior dentro de parámetros.",
                    }
                )
    return pd.DataFrame(rows).sort_values("fecha").reset_index(drop=True)


def generate_notes(
    evolutions_df: pd.DataFrame, admissions_df: pd.DataFrame
) -> pd.DataFrame:
    # Notas médicas basadas en la evolución
    adm_service = {
        row["admission_id"]: row["servicio"] for _, row in admissions_df.iterrows()
    }
    motivos = {
        row["admission_id"]: row["motivo_ingreso"]
        for _, row in admissions_df.iterrows()
    }
    rows = []
    for _, e in evolutions_df.iterrows():
        a_id = e["admission_id"]
        service = adm_service.get(a_id, "Medicina Interna")
        motivo = motivos.get(a_id, "Proceso infeccioso")
        texto = (
            f"Servicio: {service}. Motivo: {motivo}. "
            f"El paciente se encuentra {e['estado_clinico'].lower()}. "
            "Se mantiene monitorización, hidratación y pauta analgésica. "
            "Plan: continuar tratamiento actual y reevaluar en 24h."
        )
        rows.append(
            {
                "note_id": uid("N"),
                "admission_id": a_id,
                "fecha": e["fecha"],
                "texto": texto,
            }
        )
    return pd.DataFrame(rows).sort_values("fecha").reset_index(drop=True)


def generate_discharges(
    admissions_df: pd.DataFrame, evolutions_df: pd.DataFrame, meds_df: pd.DataFrame
) -> pd.DataFrame:
    # Resumen y recomendaciones al alta basados en tendencia y tratamientos activos al final
    last_state = (
        evolutions_df.sort_values("fecha")
        .groupby("admission_id")["estado_clinico"]
        .last()
    )
    last_meds = (
        meds_df.sort_values("fecha_fin").groupby("admission_id").tail(3)
    )  # últimas 3
    meds_by_adm = last_meds.groupby("admission_id")["farmaco"].apply(
        lambda s: ", ".join(sorted(set(s)))
    )
    rows = []
    for _, a in admissions_df.iterrows():
        a_id = a["admission_id"]
        estado_final = last_state.get(a_id, "Estable")
        tratamientos_alta = meds_by_adm.get(a_id, "")
        resumen = (
            f"Ingreso por {a['motivo_ingreso'].lower()} con evolución {estado_final.lower()}."
            " Responde al tratamiento instaurado, sin complicaciones mayores."
        )
        recomendaciones = "Reposo relativo, hidratación adecuada, control de signos de alarma y seguimiento en consulta."
        rows.append(
            {
                "discharge_id": uid("D"),
                "admission_id": a_id,
                "fecha_alta": a["fecha_alta"],
                "diagnostico_final": a["motivo_ingreso"],
                "evolucion_resumida": resumen,
                "tratamientos_alta": tratamientos_alta,
                "pronostico": "Favorable"
                if estado_final in ["Mejoría", "Estable"]
                else "Reservado",
                "recomendaciones": recomendaciones,
            }
        )
    return pd.DataFrame(rows).sort_values("fecha_alta").reset_index(drop=True)


# -------------------------- Persistencia --------------------------


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csvs(dfs: Dict[str, pd.DataFrame], outdir: str) -> None:
    ensure_output_dir(outdir)
    for name, df in dfs.items():
        fp = os.path.join(outdir, f"{name}.csv")
        df.to_csv(fp, index=False)


def save_sqlite(dfs: Dict[str, pd.DataFrame], fp: str) -> None:
    ensure_output_dir(os.path.dirname(fp) or ".")
    con = sqlite3.connect(fp)
    try:
        for name, df in dfs.items():
            df.to_sql(name, con, if_exists="replace", index=False)
    finally:
        con.close()


# -------------------------- Main --------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generador de dataset clínico sintético (hospitalización)."
    )
    parser.add_argument(
        "--patients",
        type=int,
        default=100,
        help="Número de pacientes a generar (por defecto: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="Semilla aleatoria (por defecto: 1337)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "sqlite", "both"],
        default="csv",
        help="Formato de salida (csv, sqlite o both). Por defecto: csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="output",
        help="Directorio/base de salida (por defecto: output)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Generación
    patients = generate_patients(args.patients)
    admissions = generate_admissions(patients)
    evolutions = generate_evolutions(admissions)
    vitals = generate_vitals(admissions)
    labs = generate_labs(admissions)
    meds = generate_meds(admissions)
    procedures = generate_procedures(admissions)
    notes = generate_notes(evolutions, admissions)
    discharges = generate_discharges(admissions, evolutions, meds)

    dfs = {
        "patients": patients,
        "admissions": admissions,
        "evolutions": evolutions,
        "vitals": vitals,
        "labs": labs,
        "meds": meds,
        "procedures": procedures,
        "notes": notes,
        "discharges": discharges,
    }

    # Persistencia
    if args.format in ("csv", "both"):
        save_csvs(dfs, args.outdir)
    if args.format in ("sqlite", "both"):
        db_path = os.path.join(args.outdir, "synthetic_ehr.sqlite")
        save_sqlite(dfs, db_path)

    # Resumen
    total_adm = len(admissions)
    total_evo = len(evolutions)
    total_notes = len(notes)
    print(
        f"OK: Generados {len(patients)} pacientes, {total_adm} ingresos, {total_evo} evoluciones y {total_notes} notas."
    )
    print(f"Salida en: {os.path.abspath(args.outdir)}")
    if args.format in ("sqlite", "both"):
        print(
            f"Base SQLite: {os.path.abspath(os.path.join(args.outdir, 'synthetic_ehr.sqlite'))}"
        )


if __name__ == "__main__":
    main()
