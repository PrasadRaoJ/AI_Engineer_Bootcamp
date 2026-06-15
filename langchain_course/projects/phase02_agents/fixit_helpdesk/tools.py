"""
FixIt IT Helpdesk Agent
tools.py: Tool definitions + mock employee/device data.
"""
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from schemas import Context

# ── Mock data ─────────────────────────────────────────────────────────────────

EMPLOYEES = {
    "EMP001": {"name": "Priya Sharma",  "department": "Finance",     "email": "priya@fixit.co",  "devices": ["LP001"]},
    "EMP002": {"name": "Ravi Kumar",    "department": "Engineering", "email": "ravi@fixit.co",   "devices": ["LP002"]},
    "EMP003": {"name": "Anita Rao",     "department": "Marketing",   "email": "anita@fixit.co",  "devices": ["LP003"]},
    "EMP099": {"name": "IT Admin",      "department": "IT",          "email": "admin@fixit.co",  "devices": []},
}

DEVICES = {
    "LP001": {"employee_id": "EMP001", "model": "Dell XPS 15",   "os": "Windows 11"},
    "LP002": {"employee_id": "EMP002", "model": "ThinkPad X1",   "os": "Ubuntu 22.04"},
    "LP003": {"employee_id": "EMP003", "model": "MacBook Pro",   "os": "macOS"},
}

TICKETS = {
    "TK001": {"employee_id": "EMP001", "issue": "Password expired",                  "status": "resolved", "priority": "medium", "created_at": "2026-06-10", "resolved_at": "2026-06-10"},
    "TK002": {"employee_id": "EMP002", "issue": "VPN disconnects every 10 minutes",  "status": "open",     "priority": "high",   "created_at": "2026-06-14", "resolved_at": None},
    "TK003": {"employee_id": "EMP003", "issue": "Need Figma installed",              "status": "resolved", "priority": "low",    "created_at": "2026-06-12", "resolved_at": "2026-06-12"},
}

APPROVED_SOFTWARE = {
    "Chrome", "Slack", "Zoom", "VS Code", "IntelliJ IDEA",
    "Figma", "Postman", "Docker Desktop", "1Password", "Notion",
}

SLA_HOURS = {"critical": 2, "high": 4, "medium": 24, "low": 72}

VPN_FIXES = {
    "Ubuntu 22.04": [
        "1. Restart NetworkManager: sudo systemctl restart NetworkManager",
        "2. Fix MTU (most common fix): sudo ip link set dev tun0 mtu 1200",
        "3. Reinstall OpenVPN: sudo apt reinstall openvpn network-manager-openvpn",
    ],
    "Windows 11": [
        "1. Flush DNS: ipconfig /flushdns then reconnect",
        "2. Reinstall VPN adapter in Device Manager → Network Adapters",
        "3. Set DNS 8.8.8.8 on the VPN network adapter",
    ],
    "macOS": [
        "1. Remove and re-import VPN profile: System Settings → VPN → delete and re-add",
        "2. Delete stale Keychain entries: search 'Cisco VPN' in Keychain Access",
        "3. Update macOS — VPN keepalive issues fixed in 14.4+",
    ],
}

_ticket_counter = [4]   # TK001-TK003 used in mock data
_access_counter = [1]

def _next_tid() -> str:
    tid = f"TK{_ticket_counter[0]:03d}"
    _ticket_counter[0] += 1
    return tid

def _next_acc() -> str:
    ref = f"ACC{_access_counter[0]:03d}"
    _access_counter[0] += 1
    return ref


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def reset_password(employee_id: str) -> str:
    """Reset the password for an employee and return a temporary one."""
    emp = EMPLOYEES.get(employee_id)
    if not emp:
        return f"Employee {employee_id} not found."
    temp = f"TempPass#{employee_id}#2026!"
    return (
        f"Password reset for {emp['name']} ({employee_id}).\n"
        f"Temporary password: {temp}\n"
        f"Expires in 24 hours — employee must change on first login.\n"
        f"Reset link also sent to {emp['email']}."
    )


@tool
def check_vpn_config(employee_id: str, runtime: ToolRuntime[Context]) -> str:
    """Check the VPN configuration status for an employee and suggest OS-specific fix steps."""
    emp = EMPLOYEES.get(employee_id)
    if not emp:
        return f"Employee {employee_id} not found."

    # Read device history from LTM store for OS-specific advice
    item = runtime.store.get(("employees", employee_id), "device_history")
    if item:
        os_name = item.value.get("os", "Unknown")
        device  = item.value.get("laptop", "Unknown device")
        fixes   = VPN_FIXES.get(os_name, VPN_FIXES["Ubuntu 22.04"])
        return (
            f"VPN config for {emp['name']} — Device: {device} | OS: {os_name}\n"
            f"Status: VPN client v2.6 installed, last connected 2 hours ago.\n"
            f"Suggested fixes for {os_name}:\n" + "\n".join(fixes)
        )

    # No device history saved — generic advice
    return (
        f"VPN config for {emp['name']}: client installed, last connected 2 hours ago.\n"
        "Generic fix steps:\n"
        "1. Restart the VPN client and reconnect\n"
        "2. Check network connection — try on a different network\n"
        "3. Update VPN client to latest version\n"
        "4. If issue persists, create a ticket for further investigation."
    )


@tool
def install_software(employee_id: str, software_name: str) -> str:
    """Install approved software for an employee. Rejected if not in the approved catalogue."""
    emp = EMPLOYEES.get(employee_id)
    if not emp:
        return f"Employee {employee_id} not found."
    # Case-insensitive check against approved list
    matched = next((s for s in APPROVED_SOFTWARE if s.lower() == software_name.lower()), None)
    if not matched:
        return (
            f"'{software_name}' is not in the approved software catalogue.\n"
            f"Approved software: {', '.join(sorted(APPROVED_SOFTWARE))}.\n"
            f"Submit a software request ticket for unlisted tools."
        )
    return (
        f"Installing {matched} for {emp['name']} ({employee_id})...\n"
        f"Deployment queued — will complete within 15 minutes.\n"
        f"A confirmation email will be sent to {emp['email']}."
    )


@tool
def check_ticket_status(ticket_id: str) -> str:
    """Get the current status and last update for a support ticket."""
    tk = TICKETS.get(ticket_id.upper())
    if not tk:
        return f"Ticket {ticket_id} not found."
    emp = EMPLOYEES.get(tk["employee_id"], {})
    resolved = f"Resolved: {tk['resolved_at']}" if tk["resolved_at"] else "Not yet resolved"
    return (
        f"Ticket {ticket_id.upper()}\n"
        f"  Employee : {emp.get('name', tk['employee_id'])}\n"
        f"  Issue    : {tk['issue']}\n"
        f"  Status   : {tk['status']}\n"
        f"  Priority : {tk['priority']}\n"
        f"  Created  : {tk['created_at']}\n"
        f"  {resolved}"
    )


@tool
def create_ticket(employee_id: str, issue: str, priority: str) -> str:
    """Create a new IT support ticket. priority: low / medium / high / critical."""
    emp = EMPLOYEES.get(employee_id)
    if not emp:
        return f"Employee {employee_id} not found."
    priority = priority.lower()
    sla = SLA_HOURS.get(priority, SLA_HOURS["medium"])
    tid = _next_tid()
    TICKETS[tid] = {
        "employee_id": employee_id,
        "issue": issue,
        "status": "open",
        "priority": priority,
        "created_at": "2026-06-15",
        "resolved_at": None,
    }
    return (
        f"Ticket created — ID: {tid}\n"
        f"  Employee : {emp['name']} ({employee_id})\n"
        f"  Issue    : {issue}\n"
        f"  Priority : {priority}\n"
        f"  SLA      : Response within {sla} hour(s)\n"
        f"  A confirmation has been sent to {emp['email']}."
    )


@tool
def grant_admin_access(employee_id: str, resource: str, duration_hours: int) -> str:
    """Grant temporary admin access to a resource for an employee. Requires HITL approval."""
    emp = EMPLOYEES.get(employee_id)
    if not emp:
        return f"Employee {employee_id} not found."
    ref = _next_acc()
    return (
        f"Admin access granted — Ref: {ref}\n"
        f"  Employee : {emp['name']} ({employee_id})\n"
        f"  Resource : {resource}\n"
        f"  Duration : {duration_hours} hour(s)\n"
        f"  Access auto-revokes after {duration_hours}h. Logged in audit trail."
    )


@tool
def reset_mfa(employee_id: str) -> str:
    """Reset MFA for an employee and return a new QR code setup link. Requires HITL approval."""
    emp = EMPLOYEES.get(employee_id)
    if not emp:
        return f"Employee {employee_id} not found."
    return (
        f"MFA reset for {emp['name']} ({employee_id}).\n"
        f"Setup link (expires in 15 mins): https://auth.fixit.co/mfa-setup?token=MFA-{employee_id}-2026\n"
        f"Employee must scan the QR code before the link expires."
    )


@tool
def wipe_device(device_id: str) -> str:
    """Remotely wipe a device — all data will be erased. DESTRUCTIVE. Requires HITL approval."""
    device = DEVICES.get(device_id.upper())
    if not device:
        return f"Device {device_id} not found."
    emp = EMPLOYEES.get(device["employee_id"], {})
    return (
        f"REMOTE WIPE INITIATED — Device: {device_id.upper()}\n"
        f"  Model    : {device['model']} | OS: {device['os']}\n"
        f"  Assigned : {emp.get('name', 'Unknown')} ({device['employee_id']})\n"
        f"  All data will be erased within 5 minutes. This cannot be undone."
    )
