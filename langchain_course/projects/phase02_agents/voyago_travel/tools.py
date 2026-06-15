"""
Voyago Travel Booking Assistant
tools.py: Tool definitions + mock flight/hotel data.
"""
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from schemas import Context

# ── Mock data ─────────────────────────────────────────────────────────────────

FLIGHTS = {
    "FL001": {"origin": "HYD", "destination": "BLR", "date": "2026-06-20", "time": "08:30", "price": 4200, "airline": "IndiGo"},
    "FL002": {"origin": "HYD", "destination": "BLR", "date": "2026-06-20", "time": "14:15", "price": 3800, "airline": "Air India"},
    "FL003": {"origin": "HYD", "destination": "DEL", "date": "2026-06-22", "time": "06:00", "price": 7500, "airline": "IndiGo"},
    "FL004": {"origin": "HYD", "destination": "BOM", "date": "2026-06-22", "time": "10:00", "price": 5200, "airline": "SpiceJet"},
    "FL005": {"origin": "HYD", "destination": "MAA", "date": "2026-06-18", "time": "07:45", "price": 3200, "airline": "IndiGo"},
    "FL006": {"origin": "HYD", "destination": "MAA", "date": "2026-06-20", "time": "09:00", "price": 3500, "airline": "Air India"},
}

HOTELS = {
    "HT001": {"city": "bangalore", "name": "Voyago Grand Bangalore", "near": "airport",         "price_per_night": 4500},
    "HT002": {"city": "bangalore", "name": "City Inn MG Road",        "near": "mg road",        "price_per_night": 2800},
    "HT003": {"city": "mumbai",    "name": "BKC Business Suites",     "near": "bkc",            "price_per_night": 6200},
    "HT004": {"city": "mumbai",    "name": "Voyago Mumbai Airport",   "near": "airport",        "price_per_night": 5100},
    "HT005": {"city": "delhi",     "name": "Capital Suites Delhi",    "near": "airport",        "price_per_night": 5800},
    "HT006": {"city": "delhi",     "name": "Connaught Place Inn",     "near": "connaught place","price_per_night": 4200},
}

BOOKINGS = {}  # populated at runtime: booking_id → booking dict

_counter = [1]

def _next_bid() -> str:
    bid = f"BK{_counter[0]:03d}"
    _counter[0] += 1
    return bid

_CITY_CODES = {
    "hyderabad": "HYD", "bangalore": "BLR", "bengaluru": "BLR",
    "delhi": "DEL", "new delhi": "DEL",
    "mumbai": "BOM", "bombay": "BOM",
    "chennai": "MAA", "madras": "MAA",
}

def _code(city: str) -> str:
    return _CITY_CODES.get(city.lower().strip(), city.upper().strip())


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search available flights. origin/destination: city name or code (HYD/BLR/DEL/BOM/MAA).
    date must be YYYY-MM-DD format."""
    orig, dest = _code(origin), _code(destination)
    matches = [
        f"  {fid}: {f['airline']} | {f['time']} | ₹{f['price']}"
        for fid, f in FLIGHTS.items()
        if f["origin"] == orig and f["destination"] == dest and f["date"] == date
    ]
    if not matches:
        return f"No flights found from {orig} to {dest} on {date}."
    return f"Flights from {orig} to {dest} on {date}:\n" + "\n".join(matches)


@tool
def book_flight(flight_id: str, passenger_name: str, seat_pref: str) -> str:
    """Book a flight by flight ID. seat_pref: window / aisle / middle."""
    fl = FLIGHTS.get(flight_id.upper())
    if not fl:
        return f"Flight {flight_id} not found."
    bid = _next_bid()
    BOOKINGS[bid] = {
        "type": "flight", "flight_id": flight_id, "passenger": passenger_name,
        "seat": seat_pref, "amount": fl["price"],
        "details": f"{fl['origin']} → {fl['destination']}, {fl['date']}, {fl['time']}",
    }
    return (
        f"Flight booked! Booking ID: {bid} | "
        f"{fl['origin']} → {fl['destination']} on {fl['date']} at {fl['time']} | "
        f"Seat: {seat_pref} | ₹{fl['price']}"
    )


@tool
def cancel_booking(booking_id: str) -> str:
    """Cancel an existing flight or hotel booking by booking ID."""
    b = BOOKINGS.pop(booking_id.upper(), None)
    if not b:
        return f"Booking {booking_id} not found."
    refund = b["amount"] * 0.9
    return f"Booking {booking_id} cancelled. Refund of ₹{refund:.0f} in 3-5 business days."


@tool
def search_hotels(city: str, check_in: str, check_out: str, near: str = "") -> str:
    """Search hotels in a city. near: landmark hint like 'airport', 'BKC', 'MG Road' (optional)."""
    city_norm = _CITY_CODES.get(city.lower().strip(), city.upper().strip())
    # map codes back to city names for HOTELS dict
    code_to_city = {"BLR": "bangalore", "BOM": "mumbai", "DEL": "delhi", "MAA": "chennai", "HYD": "hyderabad"}
    city_norm = code_to_city.get(city_norm, city_norm)

    matches = []
    for hid, h in HOTELS.items():
        if h["city"] != city_norm:
            continue
        if near and near.lower() not in h["near"]:
            continue
        matches.append(f"  {hid}: {h['name']} | Near: {h['near']} | ₹{h['price_per_night']}/night")

    if not matches:
        qualifier = f" near {near}" if near else ""
        return f"No hotels found in {city_norm.title()}{qualifier}."
    header = f"Hotels in {city_norm.title()}" + (f" near {near}" if near else "") + f" ({check_in} → {check_out}):"
    return header + "\n" + "\n".join(matches)


@tool
def book_hotel(hotel_id: str, guest_name: str, nights: int) -> str:
    """Book a hotel room for a guest for the given number of nights."""
    h = HOTELS.get(hotel_id.upper())
    if not h:
        return f"Hotel {hotel_id} not found."
    bid = _next_bid()
    total = h["price_per_night"] * nights
    BOOKINGS[bid] = {
        "type": "hotel", "hotel_id": hotel_id, "guest": guest_name,
        "nights": nights, "amount": total,
        "details": f"{h['name']}, {nights} night(s), ₹{total}",
    }
    return (
        f"Hotel booked! Booking ID: {bid} | {h['name']} | "
        f"{nights} night(s) | Total: ₹{total}"
    )


@tool
def get_my_preferences(runtime: ToolRuntime[Context]) -> str:
    """Get the current user's saved travel preferences (seat type, meal, past destinations)."""
    item = runtime.store.get(("users", runtime.context.user_id), "preferences")
    if item is None:
        return "No preferences saved yet."
    lines = [f"  {k}: {v}" for k, v in item.value.items()]
    return "Your saved preferences:\n" + "\n".join(lines)


@tool
def save_preference(key: str, value: str, runtime: ToolRuntime[Context]) -> str:
    """Save or update a travel preference for the current user. e.g. key=seat, value=window."""
    uid = runtime.context.user_id
    item = runtime.store.get(("users", uid), "preferences")
    prefs = dict(item.value) if item else {}
    prefs[key] = value
    runtime.store.put(("users", uid), "preferences", prefs)
    return f"Saved preference — {key}: {value}."
