import requests
import json

BASE_URL = "http://localhost:8000"  # sau URL-ul tău

def test_session(user_profile: dict, description: str):
    print(f"\n{'='*50}")
    print(f"TEST: {description}")
    print(f"Profil: {user_profile}")
    print('='*50)

    # 1. Start session
    res = requests.post(f"{BASE_URL}/recommendations/session/start")
    data = res.json()["data"]
    session_id = data["session_id"]
    question = data["question"]
    print(f"\nSesiune creată: {session_id}")

    step = 0
    while True:
        step += 1
        q_id = question["id"]
        q_type = question["type"]

        # Simulează răspunsul din profil
        answer = simulate_answer(user_profile, question)
        print(f"  Q{step}: {q_id} -> {answer}")

        # Trimite răspunsul
        res = requests.post(
            f"{BASE_URL}/recommendations/session/{session_id}/answer",
            json={"question_id": q_id, "answer": answer}
        )
        data = res.json()["data"]

        if data["status"] == "done":
            print(f"\nOprit după {step} întrebări")
            print(f"Top 10 rezultate:")
            for i, r in enumerate(data["results"], 1):
                print(f"  {i}. {r['name'][:50]} | {r.get('cuisine')} | {r.get('meal_type')}")
            return data["results"]

        question = data["question"]
        print(f"     H={data.get('entropy', '?'):.2f} | pool estimat scade")


def simulate_answer(profile: dict, question: dict) -> any:
    q_id = question["id"]
    q_type = question["type"]

    if q_type == "categorical":
        return profile.get(q_id) or "skip"

    if q_type == "multiselect":
        val = profile.get(q_id)
        if val is None:
            return ["any"]
        return [val] if isinstance(val, str) else val

    # Boolean — skip dacă nu e în profil
    feature = question.get("feature", q_id)
    val = profile.get(feature) if feature in profile else profile.get(q_id)
    
    if val is True:
        return "yes"
    if val is False:
        return "no"
    return "skip"  # feature absent → skip, nu no


# ── TESTE ──────────────────────────────────────────────────────
if __name__ == "__main__":
    teste = [
        (
            {"meal_type": "lunch_dinner", "cuisine": "italian", "has_pasta": True},
            "Italian pasta lunch"
        ),
        (
            {"meal_type": "dessert", "has_chocolate": True, "needs_oven": True},
            "Chocolate dessert oven"
        ),
        (
            {"meal_type": "soup", "protein_type": "chicken"},
            "Chicken soup"
        ),
        (
            {"meal_type": "lunch_dinner", "protein_type": "meatless", "cuisine": "indian", "is_spicy": True},
            "Indian spicy meatless"
        ),
        (
            {"meal_type": "lunch_dinner", "is_quick": True, "protein_type": "beef_pork"},
            "Quick beef"
        ),
    ]

    for profil, descriere in teste:
        rezultate = test_session(profil, descriere)
        
        # Verifică dacă rezultatele sunt relevante
        relevante = sum(
            1 for r in rezultate
            if all(
                r.get(k) == v
                for k, v in profil.items()
                if k not in {"is_quick", "has_pasta", "has_chocolate", "needs_oven", "is_spicy"}
            )
        )
        print(f"\nRelevante în top 10: {relevante}/10")