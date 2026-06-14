from langchain_core.tools import tool


SLOTS = {
    "2026-06-15": {"Dr. Prasad (General Physician)": ["10:00 AM", "11:30 AM", "3:00 PM"],
                   "Dr. Anusha (Cardiologist)": ["9:00 AM", "2:00 PM"]},
    "2026-06-16": {"Dr. Prasad (General Physician)": ["9:30 AM", "1:00 PM"],
                   "Dr. Anusha (Cardiologist)": ["11:00 AM", "4:00 PM"]},
}

DOCTORS = {
    "Dr. Prasad": {"specialization": "General Physician", "experience": "15 years", "fee": "₹500"},
    "Dr. Anusha": {"specialization": "Cardiologist", "experience": "20 years", "fee": "₹1200"},
}

APPOINTMENTS = {
    "APT001": {"patient": "Ravi Kumar", "doctor": "Dr. Prasad", "date": "2026-06-15", "time": "10:00 AM"},
}


@tool
def check_availability(date: str, specialization: str) -> str:
    """Returns available appointment slots for a given date and doctor specialization."""
    day = SLOTS.get(date, {})
    matches = {doc: slots for doc, slots in day.items() if specialization.lower() in doc.lower()}
    if not matches:
        return f"No availability found for {specialization} on {date}."
    result = []
    for doc, slots in matches.items():
        result.append(f"{doc}: {', '.join(slots)}")
    return "\n".join(result)


@tool
def book_appointment(patient_name: str, doctor: str, date: str, time: str) -> str:
    """Books an appointment for a patient with a doctor at a given date and time."""
    apt_id = f"APT{len(APPOINTMENTS)+1:03d}"
    APPOINTMENTS[apt_id] = {"patient": patient_name, "doctor": doctor, "date": date, "time": time}
    return f"Appointment confirmed! ID: {apt_id} | {doctor} on {date} at {time}. Please arrive 10 mins early."


@tool
def cancel_appointment(appointment_id: str) -> str:
    """Cancels an existing appointment given its appointment ID."""
    if appointment_id in APPOINTMENTS:
        apt = APPOINTMENTS.pop(appointment_id)
        return f"Appointment {appointment_id} for {apt['patient']} with {apt['doctor']} on {apt['date']} has been cancelled."
    return f"Appointment {appointment_id} not found."


@tool
def get_doctor_info(doctor_name: str) -> str:
    """Returns profile information for a doctor including specialization, experience, and fee."""
    for name, info in DOCTORS.items():
        if doctor_name.lower() in name.lower():
            return f"{name} — {info['specialization']} | {info['experience']} experience | Consultation fee: {info['fee']}"
    return f"Doctor '{doctor_name}' not found."


ALL_TOOLS = [check_availability, book_appointment, cancel_appointment, get_doctor_info]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
