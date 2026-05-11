# Recipe Match

Recipe Match este o aplicatie full-stack pentru recomandari de retete. Frontend-ul este o aplicatie mobila construita cu Expo si React Native, iar backend-ul este un API FastAPI care foloseste Supabase pentru autentificare, baza de date si acces la date.

Aplicatia ajuta utilizatorul sa gaseasca retete potrivite preferintelor sale prin trei mecanisme principale:

- un flow interactiv de recomandare bazat pe intrebari si un motor Bayesian;
- un feed personalizat `For You`, bazat pe retetele salvate;
- un `Ingredient Explorer`, care recomanda ingrediente compatibile si retete pornind de la ingrediente selectate.

## Cuprins

- [Functionalitati](#functionalitati)
- [Tehnologii](#tehnologii)
- [Structura proiectului](#structura-proiectului)
- [Arhitectura](#arhitectura)
- [Fluxuri principale](#fluxuri-principale)
- [Backend API](#backend-api)
- [Baza de date](#baza-de-date)
- [Motorul de recomandare](#motorul-de-recomandare)
- [Ingredient Explorer](#ingredient-explorer)
- [Configurare si rulare](#configurare-si-rulare)
- [Scripturi utile](#scripturi-utile)
- [Observatii de productie](#observatii-de-productie)

## Functionalitati

Aplicatia include:

- autentificare si inregistrare cu Supabase Auth;
- profil alimentar pentru fiecare utilizator;
- restrictii alimentare: vegetarian, vegan, gluten-free, dairy-free;
- ingrediente excluse de utilizator;
- recomandari interactive prin intrebari;
- recomandari personalizate pe baza retetelor salvate;
- cautare si explorare de ingrediente;
- retete similare semantic folosind embeddings si pgvector;
- salvarea retetelor;
- organizarea retetelor salvate in colectii;
- pagina de detaliu pentru retete;
- generare lista de cumparaturi din ingredientele retetei;
- suport pentru interactiuni de tip `view`, `like`, `save`, `cook`, `skip`.

## Tehnologii

### Frontend

- Expo SDK 54
- React 19
- React Native 0.81
- Expo Router pentru navigatie bazata pe fisiere
- Zustand pentru starea globala de autentificare
- AsyncStorage pentru persistarea tokenului
- React Native Reanimated si Gesture Handler pentru interactiuni mobile
- Expo Image pentru afisarea imaginilor

### Backend

- FastAPI
- Pydantic si pydantic-settings
- Supabase Auth
- Supabase PostgREST
- SlowAPI pentru rate limiting
- NumPy pentru motorul Bayesian
- sentence-transformers pentru embeddings

### Persistenta

- Supabase PostgreSQL
- Supabase Auth users
- pgvector pentru similaritate semantica
- tabele pentru retete, sesiuni de recomandare, interactiuni, colectii, retete salvate si graf de ingrediente

## Structura proiectului

```text
recipe_match/
  backend/
    app/
      main.py
      config.py
      database.py
      middleware/
        auth.py
      models/
        schemas.py
      recommender/
        embeddings.py
        engine.py
        filters.py
        semantic_rerank.py
      routers/
        auth.py
        collections.py
        explorer.py
        foryou.py
        recipes.py
        recommendations.py
        saved.py
    migrations/
      001_recipe_features.sql
      002_recipe_embeddings.sql
      003_ingredient_explorer.sql
    scripts/
      precompute_embeddings.py
      precompute_ingredient_graph.py
    testari/

  frontend/
    recipe-match/
      app/
        _layout.tsx
        (auth)/
        (tabs)/
        recipe/[id].tsx
      components/
      constants/
      hooks/
      services/
      store/
      assets/
```

## Arhitectura

### Frontend

Frontend-ul este organizat in jurul Expo Router:

- `app/_layout.tsx` initializeaza autentificarea si decide daca utilizatorul ajunge in zona autentificata sau in ecranele de login/register.
- `app/(auth)` contine ecranele de login si register.
- `app/(tabs)` contine tab-urile principale ale aplicatiei:
  - `index.tsx` pentru `For You`;
  - `find.tsx` pentru flow-ul de recomandare;
  - `saved.tsx` pentru retete salvate si colectii;
  - `explorer.tsx` pentru explorarea ingredientelor;
  - `profile.tsx` pentru profil.
- `app/recipe/[id].tsx` afiseaza detaliile unei retete.
- `services/api.ts` si `services/explorerApi.ts` contin clientii HTTP.
- `store/authStore.ts` pastreaza tokenul, profilul utilizatorului si actiunile de autentificare.

Tokenul Supabase este salvat local in AsyncStorage sub cheia `auth_token` si este trimis catre backend in headerul:

```http
Authorization: Bearer <token>
```

### Backend

Backend-ul este un API FastAPI modular:

- `main.py` creeaza aplicatia, configureaza CORS, rate limiting si include routerele.
- `config.py` citeste variabilele de mediu din `backend/.env`.
- `database.py` creeaza clientul Supabase anon si clientul service-role.
- `middleware/auth.py` valideaza JWT-ul Supabase si extrage `user_id`.
- `models/schemas.py` defineste modelele Pydantic pentru requesturi si raspunsuri.
- `routers/` contine endpoint-urile pe domenii functionale.
- `recommender/engine.py` contine motorul Bayesian.
- `recommender/semantic_rerank.py` combina scoruri semantice cu semnale din raspunsuri.
- `recommender/filters.py` filtreaza retetele dupa ingrediente excluse.
- `recommender/embeddings.py` construieste textul retetelor si genereaza embeddings.

La pornire, backend-ul incarca retetele din Supabase in memorie pentru motorul Bayesian si calculeaza ponderi de informatie pentru intrebari.

## Fluxuri principale

### Autentificare

1. Utilizatorul creeaza cont sau se logheaza.
2. Backend-ul foloseste Supabase Auth.
3. La login/register, backend-ul returneaza tokenul si profilul utilizatorului.
4. Frontend-ul salveaza tokenul in AsyncStorage.
5. La pornirea aplicatiei, frontend-ul valideaza tokenul prin `GET /auth/me`.

Profilul alimentar este salvat in metadata-ul utilizatorului Supabase si contine:

- `is_vegetarian`
- `is_vegan`
- `is_gluten_free`
- `is_dairy_free`
- `excluded_ingredients`

### Flow-ul Find

1. Frontend-ul apeleaza `POST /recommendations/session/start`.
2. Backend-ul filtreaza retetele dupa profilul alimentar.
3. Se creeaza o sesiune in memorie.
4. Utilizatorul raspunde la intrebari.
5. Backend-ul actualizeaza probabilitatile Bayesiene.
6. Backend-ul decide urmatoarea intrebare sau returneaza rezultatele.
7. Frontend-ul afiseaza retete cu `match_score`.

Intrebarile fixe sunt:

- `meal_type`
- `protein_type`
- `cuisine`

Intrebarile adaptive sunt legate de:

- gust: spicy, sweet;
- timp: quick;
- metoda de gatit: oven, stovetop, no-cook;
- ingrediente si baze culinare: pasta, rice, potato, tomato base, cream base, cheese, broth;
- categorii de ingrediente: mushroom, leafy greens, beans/legumes, fruit, nuts, chocolate;
- alte semnale: tortilla, spicy ingredient, asian sauce.

### For You

Endpoint-ul `GET /foryou` returneaza recomandari pornind de la retetele salvate de utilizator.

Backend-ul:

1. citeste retetele salvate din `saved_recipes`;
2. construieste un profil simplu din meal type, cuisine si flaguri precum quick/spicy/sweet;
3. exclude retetele deja salvate;
4. sorteaza retetele candidate dupa scorul de potrivire.

### Retete salvate si colectii

Utilizatorul poate salva retete si le poate organiza in colectii. Daca salveaza o reteta fara sa aleaga o colectie, backend-ul creeaza sau refoloseste automat colectia implicita `Saved`.

### Ingredient Explorer

Ingredient Explorer permite pornirea de la un ingredient si extinderea listei cu ingrediente compatibile. Sugestiile sunt calculate folosind:

- frecventa ingredientelor in retetele care contin ingredientele selectate;
- scoruri PPMI din tabelul `ingredient_graph`;
- numarul de retete in care apare fiecare ingredient.

## Backend API

Toate raspunsurile API sunt invelite intr-un format comun:

```json
{
  "data": {},
  "error": null
}
```

In caz de eroare:

```json
{
  "data": null,
  "error": "Mesaj de eroare"
}
```

### Auth

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| POST | `/auth/register` | Creeaza utilizator si profil alimentar |
| POST | `/auth/login` | Autentifica utilizatorul |
| POST | `/auth/logout` | Logout |
| GET | `/auth/me` | Returneaza profilul curent |
| PATCH | `/auth/me` | Actualizeaza profilul alimentar |

### Recipes

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| GET | `/recipes/ingredients` | Sugestii de ingrediente pentru cautare |
| GET | `/recipes/{recipe_id}` | Detalii reteta |
| GET | `/recipes/{recipe_id}/shopping-list` | Lista de cumparaturi |
| GET | `/recipes/{recipe_id}/similar` | Retete similare semantic |

### Saved

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| POST | `/saved` | Salveaza o reteta |
| GET | `/saved` | Listeaza toate retetele salvate |
| DELETE | `/saved/{recipe_id}` | Sterge o reteta salvata |
| GET | `/saved/collections/{collection_id}` | Listeaza retetele dintr-o colectie |

### Collections

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| GET | `/collections` | Listeaza colectiile utilizatorului |
| POST | `/collections` | Creeaza o colectie |
| DELETE | `/collections/{collection_id}` | Sterge o colectie |

### For You

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| GET | `/foryou` | Recomandari personalizate din istoricul salvarilor |

### Recommendations

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| POST | `/recommendations/session/start` | Porneste o sesiune de recomandare |
| POST | `/recommendations/session/{session_id}/answer` | Trimite raspunsul la o intrebare |
| GET | `/recommendations/session/{session_id}/results` | Returneaza rezultatele sesiunii |
| POST | `/recommendations/interaction` | Inregistreaza o interactiune cu o reteta |

### Explorer

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| POST | `/explorer/start` | Porneste explorarea de la un ingredient |
| POST | `/explorer/expand` | Sugereaza ingrediente noi pe baza celor selectate |
| POST | `/explorer/recommend` | Recomanda retete pe baza ingredientelor selectate |
| GET | `/explorer/search` | Cauta ingrediente |

## Baza de date

Baza de date este Supabase PostgreSQL. Autentificarea este gestionata de Supabase Auth, iar tabelele aplicatiei sunt in schema `public`.

### `auth.users`

Tabel gestionat de Supabase Auth. Aplicatia il foloseste pentru identitatea utilizatorilor.

Datele alimentare ale utilizatorului sunt tinute in `user_metadata`.

| Camp metadata | Tip | Descriere |
| --- | --- | --- |
| `is_vegetarian` | boolean | Utilizator vegetarian |
| `is_vegan` | boolean | Utilizator vegan |
| `is_gluten_free` | boolean | Evita glutenul |
| `is_dairy_free` | boolean | Evita lactatele |
| `excluded_ingredients` | text[] / list | Ingrediente excluse manual |

### `recipes`

Tabelul principal al aplicatiei. Contine datele retetelor, campuri pentru afisare, campuri pentru filtrare si feature-uri folosite de motorul de recomandare.

#### Campuri de baza

| Coloana | Tip estimat | Descriere |
| --- | --- | --- |
| `id` | integer | Identificator reteta |
| `name` | text | Numele retetei |
| `description` | text | Descriere scurta |
| `image_url` | text | URL imagine |
| `prep_time` | text | Timp de pregatire in format text |
| `cook_time` | text | Timp de gatire in format text |
| `total_time` | text | Timp total in format text |
| `total_minutes` | double precision | Timp total numeric, folosit la sortari/filtre |
| `servings` | integer | Numar portii |
| `ingredients` | text | Ingrediente brute |
| `ingredients_clean` | json/text[] | Ingrediente normalizate |
| `ingredients_clean_str` | text | Ingrediente normalizate ca string, folosit in Explorer |
| `directions` | text | Pasi de preparare |

#### Campuri categorice

| Coloana | Tip estimat | Descriere |
| --- | --- | --- |
| `meal_type` | text | Tipul mesei: breakfast/lunch/dinner etc. |
| `protein_type` | text | Proteina dominanta |
| `cuisine` | text | Bucatarie sau stil culinar |

#### Campuri pentru restrictii si badge-uri

| Coloana | Tip | Descriere |
| --- | --- | --- |
| `is_vegetarian` | boolean | Reteta vegetariana |
| `is_vegan` | boolean | Reteta vegana |
| `is_gluten_free` | boolean | Reteta fara gluten |
| `is_dairy_free` | boolean | Reteta fara lactate |
| `is_nut_free` | boolean | Reteta fara nuci/alune |
| `is_quick` | boolean | Reteta rapida |
| `is_spicy` | boolean | Reteta picanta |
| `is_sweet` | boolean | Reteta dulce |
| `needs_oven` | boolean | Necesita cuptor |
| `needs_stovetop` | boolean | Necesita aragaz/plita |
| `is_no_cook` | boolean | Nu necesita gatire |

#### Feature-uri de ingrediente

Aceste coloane sunt adaugate prin `backend/migrations/001_recipe_features.sql` si sunt folosite de motorul Bayesian.

| Coloana | Tip | Descriere |
| --- | --- | --- |
| `has_pasta` | boolean | Contine paste |
| `has_rice` | boolean | Contine orez |
| `has_potato` | boolean | Contine cartof |
| `has_tomato_base` | boolean | Are baza de rosii |
| `has_cream_base` | boolean | Are baza cremoasa |
| `has_cheese` | boolean | Contine branza |
| `has_broth_base` | boolean | Are baza de supa/broth |
| `has_mushroom` | boolean | Contine ciuperci |
| `has_leafy_greens` | boolean | Contine verdeturi/frunze |
| `has_beans_legumes` | boolean | Contine fasole/leguminoase |
| `has_fruit` | boolean | Contine fructe |
| `has_nuts` | boolean | Contine nuci/alune |
| `has_chocolate` | boolean | Contine ciocolata |
| `has_tortilla` | boolean | Contine tortilla |
| `has_spicy_ingredient` | boolean | Contine ingrediente picante |
| `has_asian_sauce` | boolean | Contine sosuri asiatice |

#### Embeddings

Adaugate prin `backend/migrations/002_recipe_embeddings.sql`.

| Coloana | Tip | Descriere |
| --- | --- | --- |
| `embedding` | `vector(384)` | Embedding semantic generat cu `sentence-transformers/all-MiniLM-L6-v2` |

Index:

```sql
CREATE INDEX IF NOT EXISTS recipes_embedding_ivfflat_idx
    ON public.recipes
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

Functie pentru retete similare:

```sql
public.match_similar_recipes(
    target_recipe_id integer,
    match_count integer DEFAULT 20
)
```

Returneaza retete ordonate dupa similaritate cosine fata de embedding-ul retetei tinta.

### `collections`

Tabel pentru colectiile utilizatorului.

| Coloana | Tip estimat | Descriere |
| --- | --- | --- |
| `id` | uuid | Identificator colectie |
| `user_id` | uuid | Utilizatorul proprietar |
| `name` | text | Numele colectiei |
| `created_at` | timestamp | Data crearii |

Comportament:

- fiecare utilizator poate avea mai multe colectii;
- backend-ul creeaza automat colectia `Saved` daca utilizatorul salveaza o reteta fara colectie.

### `saved_recipes`

Tabel pentru retetele salvate.

| Coloana | Tip estimat | Descriere |
| --- | --- | --- |
| `id` | uuid | Identificator rand |
| `user_id` | uuid | Utilizatorul care a salvat reteta |
| `recipe_id` | integer | Reteta salvata |
| `collection_id` | uuid/null | Colectia in care este salvata |
| `saved_at` | timestamp | Data salvarii |

Relatii:

- `user_id` refera utilizatorul Supabase;
- `recipe_id` refera `recipes.id`;
- `collection_id` refera `collections.id`.

### `recommendation_sessions`

Creat prin `backend/migrations/001_recipe_features.sql`. Pastreaza sumarul sesiunilor de recomandare.

| Coloana | Tip | Descriere |
| --- | --- | --- |
| `id` | uuid | Identificator sesiune |
| `user_id` | uuid | Utilizatorul sesiunii, referinta la `auth.users(id)` |
| `created_at` | timestamp | Momentul crearii |
| `completed_at` | timestamp/null | Momentul finalizarii |
| `answers` | jsonb | Raspunsurile date de utilizator |
| `question_order` | text[] | Ordinea intrebarilor puse |
| `top_recipe_ids` | integer[] | Retetele de top la finalul sesiunii |
| `questions_asked` | integer | Numar intrebari puse |
| `entropy_final` | double precision | Entropia finala a distributiei Bayesiene |

Observatie: sesiunea activa este tinuta in memorie, iar acest tabel pastreaza doar sumarul/persistenta.

### `recipe_interactions`

Creat prin `backend/migrations/001_recipe_features.sql`. Poate fi folosit pentru personalizare mai avansata.

| Coloana | Tip | Descriere |
| --- | --- | --- |
| `id` | uuid | Identificator interactiune |
| `user_id` | uuid | Utilizatorul, referinta la `auth.users(id)` |
| `recipe_id` | integer | Reteta, referinta la `recipes.id` |
| `interaction_type` | text | Tip interactiune: `view`, `like`, `save`, `cook`, `skip` |
| `weight` | double precision | Greutatea interactiunii |
| `created_at` | timestamp | Momentul interactiunii |

### `ingredient_graph`

Creat prin `backend/migrations/003_ingredient_explorer.sql`. Este folosit pentru recomandari de ingrediente compatibile.

| Coloana | Tip | Descriere |
| --- | --- | --- |
| `ingredient_a` | text | Ingredient sursa |
| `ingredient_b` | text | Ingredient asociat |
| `ppmi_score` | float | Scor Positive Pointwise Mutual Information |
| `co_occurrence` | integer | Numar de retete in care apar impreuna |

Cheie primara:

```sql
PRIMARY KEY (ingredient_a, ingredient_b)
```

Index:

```sql
CREATE INDEX IF NOT EXISTS ingredient_graph_a_score_idx
  ON ingredient_graph (ingredient_a, ppmi_score DESC);
```

### `ingredient_stats`

Creat prin `backend/migrations/003_ingredient_explorer.sql`. Este folosit pentru cautare si validarea ingredientelor.

| Coloana | Tip | Descriere |
| --- | --- | --- |
| `ingredient` | text | Ingredient normalizat |
| `recipe_count` | integer | Numar de retete in care apare |
| `total_recipes` | integer | Numar total de retete analizate |

Cheie primara:

```sql
PRIMARY KEY (ingredient)
```

Index:

```sql
CREATE INDEX IF NOT EXISTS ingredient_stats_recipe_count_idx
  ON ingredient_stats (recipe_count DESC);
```

## Motorul de recomandare

Motorul Bayesian este implementat in `backend/app/recommender/engine.py`.

### Date incarcate la startup

La pornirea API-ului, `main.py` incarca din `recipes` campurile necesare:

- identificare si afisare: `id`, `name`, `image_url`, `description`;
- categorii: `meal_type`, `protein_type`, `cuisine`;
- ingrediente: `ingredients`, `ingredients_clean`;
- embeddings: `embedding`;
- restrictii: `is_vegetarian`, `is_vegan`, `is_gluten_free`, `is_dairy_free`;
- preferinte: `is_spicy`, `is_sweet`, `is_quick`;
- metode de gatit: `needs_oven`, `needs_stovetop`, `is_no_cook`;
- feature-uri de ingrediente: `has_pasta`, `has_rice`, `has_potato`, `has_tomato_base`, `has_cream_base`, `has_cheese`, `has_broth_base`, `has_mushroom`, `has_leafy_greens`, `has_beans_legumes`, `has_fruit`, `has_nuts`, `has_chocolate`, `has_tortilla`, `has_spicy_ingredient`, `has_asian_sauce`.

### Cum functioneaza

1. Retetele sunt filtrate dupa restrictiile alimentare ale utilizatorului.
2. Fiecare reteta porneste cu probabilitate uniforma.
3. Pentru fiecare raspuns, motorul calculeaza likelihood-ul fiecarei retete.
4. Probabilitatile sunt actualizate in log-space pentru stabilitate numerica.
5. Urmatoarea intrebare este aleasa dupa reducerea asteptata a entropiei.
6. Sesiunea se opreste cand:
   - intrebarile fixe au fost puse;
   - entropia este suficient de mica;
   - topul este stabil;
   - s-a atins numarul maxim de intrebari.
7. Rezultatele sunt intoarse ca lista de retete cu `match_score`.

### Semantic reranking

Aplicatia poate folosi embeddings pentru retete similare si reranking semantic. Textul unei retete este construit din nume, descriere, ingrediente, meal type, cuisine si alte campuri relevante. Embedding-ul are dimensiunea 384 si este generat cu modelul `all-MiniLM-L6-v2`.

## Ingredient Explorer

`Ingredient Explorer` foloseste doua surse:

- ingredientele normalizate din `recipes.ingredients_clean`;
- graful de co-occurenta din `ingredient_graph`.

Flow:

1. Utilizatorul cauta sau selecteaza un ingredient.
2. Backend-ul verifica ingredientul in `ingredient_stats`.
3. Backend-ul returneaza ingrediente asociate din `ingredient_graph`.
4. Daca utilizatorul selecteaza mai multe ingrediente, backend-ul cauta retete care contin toate ingredientele selectate.
5. Sugestiile urmatoare sunt calculate combinand frecventa in retetele potrivite si scorul PPMI.
6. Recomandarile de retete sunt sortate dupa scor Jaccard intre ingredientele selectate si ingredientele retetei.

## Configurare si rulare

### Variabile de mediu backend

Creeaza `backend/.env` pornind de la `backend/.env.example`:

```env
SUPABASE_URL=
SUPABASE_KEY=
SUPABASE_SERVICE_KEY=
SECRET_KEY=
ENVIRONMENT=development
```

Descriere:

| Variabila | Descriere |
| --- | --- |
| `SUPABASE_URL` | URL-ul proiectului Supabase |
| `SUPABASE_KEY` | Cheia anon/publica Supabase |
| `SUPABASE_SERVICE_KEY` | Cheia service-role pentru operatii administrative |
| `SECRET_KEY` | Secret local folosit de aplicatie |
| `ENVIRONMENT` | `development` sau `production` |

Nu publica niciodata `backend/.env`.

### Pornire backend

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```http
GET http://localhost:8000/health
```

Documentatie Swagger in development:

```text
http://localhost:8000/docs
```

### Pornire frontend

```powershell
cd frontend\recipe-match
npm install
npm start
```

Comenzi disponibile:

```powershell
npm run android
npm run ios
npm run web
npm run lint
```

URL-ul backend-ului este configurat in:

```text
frontend/recipe-match/constants/api.ts
```

In development, aplicatia foloseste IP-ul LAN al masinii pe care ruleaza backend-ul, deoarece pe emulator/device `localhost` nu se refera mereu la host.

## Scripturi utile

### Precomputare embeddings

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python scripts\precompute_embeddings.py
```

Scriptul calculeaza embeddings pentru retete si le salveaza in coloana `recipes.embedding`.

### Precomputare graf ingrediente

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python scripts\precompute_ingredient_graph.py
```

Scriptul construieste:

- `ingredient_graph`
- `ingredient_stats`

pe baza ingredientelor normalizate din retete.

## Observatii de productie

- `backend/.env` contine secrete si trebuie sa ramana ignorat de Git.
- `SUPABASE_SERVICE_KEY` nu trebuie expus niciodata in frontend.
- In productie, CORS trebuie restrans la domeniul real al aplicatiei.
- Sesiunile active de recomandare sunt tinute in memorie; la restartul backend-ului, sesiunile active se pierd.
- Persistarea sesiunilor in `recommendation_sessions` este fire-and-forget si nu blocheaza raspunsul catre utilizator.
- `recipe_interactions` este pregatit pentru personalizare mai avansata, dar personalizarea principala foloseste in prezent retetele salvate.
- Pentru scalare, motorul de recomandare ar trebui mutat spre cache persistent sau job-uri separate daca numarul de retete creste mult.
- `node_modules`, mediile virtuale, cache-urile si fisierele `.env` sunt excluse prin `.gitignore`.

## Directii posibile de dezvoltare

- personalizare mai buna folosind `recipe_interactions`;
- meal planner saptamanal;
- scoring nutritional;
- liste de cumparaturi agregate pentru mai multe retete;
- feedback explicit pe recomandari;
- colectii smart generate automat;
- analytics pentru calitatea recomandarilor;
- panou admin pentru curatarea ingredientelor si retetelor.
