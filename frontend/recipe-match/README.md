# Recipe Match

Recipe Match este o aplicatie mobila construita cu Expo/React Native si un backend FastAPI care recomanda retete pe baza preferintelor utilizatorului. Aplicatia combina autentificare Supabase, profil alimentar, sesiuni interactive de recomandare, retete salvate, colectii si un feed personalizat "For You".

Proiectul are doua parti principale:

- `frontend/recipe-match`: aplicatia Expo React Native.
- `backend`: API-ul FastAPI, integrarea cu Supabase si motoarele de recomandare.

## Ce face aplicatia

Utilizatorul isi creeaza cont, isi seteaza restrictiile alimentare si poate descoperi retete in doua moduri:

1. Prin tab-ul `Find`, unde raspunde la intrebari despre tipul mesei, proteina, bucatarie si preferinte de ingrediente/gatit. Backend-ul calculeaza un top de retete potrivite.
2. Prin tab-ul `For You`, unde backend-ul recomanda retete pornind de la istoricul retetelor salvate.

Utilizatorul poate deschide o reteta, vedea ingrediente si pasi, genera o lista de cumparaturi, salva reteta si organiza retetele in colectii.

## Tehnologii

Frontend:

- Expo SDK 54
- React 19 si React Native 0.81
- Expo Router pentru routing bazat pe fisiere
- Zustand pentru starea de autentificare
- AsyncStorage pentru persistarea tokenului
- React Native Reanimated si Gesture Handler pentru cardurile swipe din fluxul de recomandare
- Expo Image pentru imagini

Backend:

- FastAPI
- Supabase Auth si Supabase PostgREST
- Pydantic pentru validare schema
- SlowAPI pentru rate limiting
- NumPy pentru motorul Bayesian
- sentence-transformers pentru embeddings semantice in `Find more like this`

Persistenta:

- Supabase Auth pentru utilizatori
- Tabele Supabase pentru `recipes`, `collections`, `saved_recipes`, `recommendation_sessions`, `recipe_interactions`

## Structura proiectului

```text
recipe_match/
  backend/
    app/
      main.py                 # Creeaza aplicatia FastAPI si include toate routerele
      config.py               # Citeste variabilele de mediu
      database.py             # Clientii Supabase anon si service-role
      middleware/auth.py      # Validare JWT Supabase
      models/schemas.py       # Modele Pydantic pentru request/response
      routers/
        auth.py               # Register, login, logout, profil utilizator
        recipes.py            # Detalii reteta si lista cumparaturi
        saved.py              # Salvare/stergere/listare retete salvate
        collections.py        # Colectii/foldere de retete
        foryou.py             # Recomandari personalizate din istoricul salvarilor
        recommendations.py    # Sesiuni de recomandare Bayesian + interactiuni
      recommender/engine.py   # Logica motorului Bayesian
      migrations/
      001_recipe_features.sql # Coloane feature + tabele pentru sesiuni/interactiuni
      002_recipe_embeddings.sql # pgvector + functie pentru retete similare
    scripts/
      precompute_embeddings.py # Calculeaza embeddings si le salveaza in Supabase
    testari/                  # Scripturi experimentale si evaluari

  frontend/recipe-match/
    app/
      _layout.tsx             # Guard global de autentificare
      (auth)/                 # Login/register
      (tabs)/                 # For You, Find, Saved
      recipe/[id].tsx         # Ecran detaliu reteta
    services/api.ts           # Client HTTP si tipuri TypeScript
    store/authStore.ts        # Starea globala de autentificare
    hooks/                    # Hook-uri pentru recomandari, saved, For You
    components/               # RecipeCard, LoadingSpinner etc.
    constants/                # Culori si API base URL
```

## Fluxul aplicatiei

### 1. Pornire si autentificare

La pornire, `app/_layout.tsx` apeleaza `useAuthStore.initialize()`. Store-ul cauta `auth_token` in AsyncStorage.

- Daca exista token, frontend-ul apeleaza `GET /auth/me`.
- Daca tokenul este valid, utilizatorul ramane in zona `(tabs)`.
- Daca tokenul lipseste sau este invalid, utilizatorul este trimis la `/(auth)/login`.

Autentificarea este gestionata in `store/authStore.ts`:

- `login()` apeleaza `POST /auth/login`, salveaza tokenul si profilul.
- `register()` apeleaza `POST /auth/register`, salveaza tokenul si profilul daca Supabase returneaza sesiune.
- `logout()` apeleaza `POST /auth/logout` si sterge tokenul local.
- `updateDietary()` actualizeaza restrictiile alimentare ale utilizatorului.

### 2. Register

Ecranul `app/(auth)/register.tsx` colecteaza:

- email
- parola si confirmare
- preferinte alimentare: vegetarian, vegan, gluten-free, dairy-free
- ingrediente excluse, adaugate prin cautare sau text liber

Backend-ul trimite aceste preferinte in `user_metadata` Supabase. Aceste valori sunt apoi folosite ca filtre dure in recomandarile de retete. Ingredientele excluse sunt comparate cu `ingredients_clean` si `ingredients`, iar retetele care contin acele ingrediente sunt eliminate din `Find` si `For You`.

### 3. Login

Ecranul `app/(auth)/login.tsx` trimite email si parola catre backend. Backend-ul foloseste Supabase Auth pentru verificare si intoarce un JWT. Frontend-ul trimite acest JWT la toate requesturile viitoare in headerul:

```http
Authorization: Bearer <token>
```

### 4. Tab-ul For You

Ecranul `app/(tabs)/index.tsx` foloseste `useForYou()`, care apeleaza `GET /foryou`.

Backend-ul:

1. Citeste retetele salvate ale utilizatorului din `saved_recipes`.
2. Daca utilizatorul nu are retete salvate, intoarce un set initial de retete.
3. Daca exista istoric, construieste un profil din retetele salvate:
   - tipuri de masa preferate
   - bucatarii preferate
   - frecventa retetelor picante, dulci si rapide
4. Cauta retete candidate care se potrivesc profilului.
5. Exclude retetele deja salvate.
6. Sorteaza candidatii dupa un scor de preferinta.

Rezultatul este afisat cu `RecipeCard`.

### 5. Tab-ul Find

Ecranul `app/(tabs)/find.tsx` foloseste hook-ul `useRecommendation()`, deci fluxul activ al aplicatiei foloseste motorul nou din `/recommendations`.

Flux:

1. Utilizatorul apasa butonul de start.
2. Frontend-ul apeleaza `POST /recommendations/session/start`.
3. Backend-ul filtreaza retetele dupa restrictiile alimentare ale utilizatorului.
4. Se creeaza o sesiune in memorie cu `session_id`.
5. Prima intrebare este returnata catre frontend.
6. Frontend-ul afiseaza intrebari fixe sub forma de chip-uri:
   - `meal_type`
   - `protein_type`
   - `cuisine`
7. Intrebarile booleene adaptive sunt afisate ca swipe card:
   - swipe dreapta = `yes`
   - swipe stanga = `no`
   - buton `Skip` = `skip`
8. Pentru fiecare raspuns, frontend-ul apeleaza `POST /recommendations/session/{session_id}/answer`.
9. Backend-ul actualizeaza probabilitatile Bayesiene si decide daca mai trebuie o intrebare sau daca poate returna rezultatele.

La final, frontend-ul afiseaza o grila cu top retete si procentul de potrivire.

### 6. Detaliu reteta

Ecranul `app/recipe/[id].tsx` apeleaza:

- `GET /recipes/{id}` pentru detalii complete
- `GET /collections` pentru lista colectiilor utilizatorului

Pe pagina de detaliu se afiseaza:

- imagine sau placeholder
- nume
- timp total
- portii
- bucatarie
- badge-uri: Quick, Vegan, Vegetarian, Gluten-free, Dairy-free, Spicy, Sweet
- descriere
- ingrediente
- pasi de preparare

Actiuni disponibile:

- salvare intr-o colectie
- stergere din salvate
- lista de cumparaturi prin `GET /recipes/{id}/shopping-list`
- share nativ

### 7. Saved si colectii

Ecranul `app/(tabs)/saved.tsx` foloseste `useSavedRecipes()`.

Hook-ul incarca in paralel:

- `GET /collections`
- `GET /saved`

Utilizatorul poate:

- vedea toate retetele salvate
- filtra dupa colectie
- crea o colectie noua cu `POST /collections`
- sterge o colectie cu `DELETE /collections/{id}`
- sterge o reteta salvata cu `DELETE /saved/{recipe_id}`

Daca utilizatorul salveaza o reteta fara colectie, backend-ul creeaza sau refoloseste automat colectia implicita `Saved`.

## Backend API

Toate raspunsurile sunt invelite in formatul:

```json
{
  "data": {},
  "error": null
}
```

Daca apare o eroare, `data` poate fi `null`, iar `error` contine mesajul.

### Auth

```http
POST /auth/register
POST /auth/login
POST /auth/logout
GET  /auth/me
PATCH /auth/me
```

`/auth/register` primeste email, parola si profil alimentar. `/auth/login` returneaza tokenul Supabase. `/auth/me` citeste profilul curent din JWT/Supabase.

### Recipes

```http
GET /recipes/{recipe_id}
GET /recipes/{recipe_id}/shopping-list
```

`/recipes/{id}` intoarce detalii complete despre reteta. `/shopping-list` sparge campul brut `ingredients` in linii curate.

### Saved

```http
POST   /saved
GET    /saved
DELETE /saved/{recipe_id}
GET    /saved/collections/{collection_id}
```

Aceste endpoint-uri cer autentificare. Salvarea foloseste `user_id` din JWT si scrie in `saved_recipes`.

### Collections

```http
GET    /collections
POST   /collections
DELETE /collections/{collection_id}
```

Colectiile sunt foldere simple pentru retete salvate.

### For You

```http
GET /foryou
```

Returneaza retete recomandate pe baza retetelor salvate.

### Similar recipes

```http
GET /recipes/{recipe_id}/similar
```

Returneaza retete similare semantic cu reteta curenta. Similaritatea foloseste embeddings `sentence-transformers/all-MiniLM-L6-v2` salvate in `recipes.embedding` prin extensia Supabase/Postgres `pgvector`.

Pentru initializare:

```bash
cd backend
.\venv\Scripts\Activate.ps1
python scripts/precompute_embeddings.py
```

### Recommendation session - motorul activ in frontend

```http
POST /recommendations/session/start
POST /recommendations/session/{session_id}/answer
GET  /recommendations/session/{session_id}/results
POST /recommendations/interaction
```

Acest motor este implementat in `backend/app/recommender/engine.py` si expus prin `backend/app/routers/recommendations.py`.

## Motorul Bayesian de recomandare

Motorul nou are urmatorii pasi:

1. La startup, `main.py` incarca retetele din Supabase in `app.state.rec_recipes`.
2. Se calculeaza ponderi de informatie pentru intrebari cu `compute_feature_mi()`.
3. La startul unei sesiuni, retetele sunt filtrate dupa profilul alimentar al utilizatorului.
4. `BayesianSession` porneste cu probabilitate uniforma pentru fiecare reteta.
5. Pentru fiecare raspuns, `compute_likelihood()` calculeaza cat de compatibila este fiecare reteta cu raspunsul.
6. Probabilitatile sunt actualizate in log-space pentru stabilitate numerica.
7. Urmatoarea intrebare este aleasa prin reducerea asteptata a entropiei.
8. Sesiunea se opreste cand:
   - au trecut intrebarile fixe,
   - entropia este sub pragul configurat,
   - probabilitatea este suficient concentrata in top 10,
   - sau s-a ajuns la `MAX_QUESTIONS`.

Intrebarile fixe sunt:

- `meal_type`
- `protein_type`
- `cuisine`

Intrebarile adaptive includ preferinte precum:

- spicy/sweet/quick
- oven/stovetop/no-cook
- pasta/rice/potato
- tomato/cream/cheese/broth
- mushroom/leafy greens/beans/fruit/nuts/chocolate
- tortilla/spicy ingredient/asian sauce

Scorul final `match_score` este un procent 0-100 calculat din ponderea raspunsurilor potrivite.

## Baza de date

Aplicatia presupune existenta tabelului `recipes` cu campuri precum:

- `id`, `name`, `description`, `image_url`
- `prep_time`, `cook_time`, `total_time`, `total_minutes`
- `servings`, `ingredients`, `ingredients_clean`, `directions`
- `meal_type`, `protein_type`, `cuisine`
- `is_vegetarian`, `is_vegan`, `is_gluten_free`, `is_dairy_free`
- `excluded_ingredients` in metadata-ul utilizatorului, pentru ingrediente care nu trebuie sa apara in recomandari
- `is_nut_free`, `is_quick`, `is_spicy`, `is_sweet`
- `needs_oven`, `needs_stovetop`, `is_no_cook`
- feature-uri de ingrediente: `has_pasta`, `has_rice`, `has_potato`, `has_tomato_base`, `has_cream_base`, `has_cheese`, `has_broth_base`, `has_mushroom`, `has_leafy_greens`, `has_beans_legumes`, `has_fruit`, `has_nuts`, `has_chocolate`, `has_tortilla`, `has_spicy_ingredient`, `has_asian_sauce`

Migratia `backend/migrations/001_recipe_features.sql` adauga feature-urile de ingrediente si creeaza:

- `recommendation_sessions`
- `recipe_interactions`

Mai sunt folosite si tabelele:

- `collections`
- `saved_recipes`

## Configurare backend

Backend-ul citeste variabilele din `backend/.env` prin `pydantic-settings`.

Variabile necesare:

```env
SUPABASE_URL=...
SUPABASE_KEY=...
SUPABASE_SERVICE_KEY=...
SECRET_KEY=...
ENVIRONMENT=development
```

`SUPABASE_KEY` este cheia anon, folosita pentru operatii de autentificare. `SUPABASE_SERVICE_KEY` este cheia service-role, folosita pentru citiri/scrieri administrative si bypass RLS.

Pornire backend:

```bash
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoint de verificare:

```http
GET http://localhost:8000/health
```

In development, documentatia FastAPI este disponibila la:

```text
http://localhost:8000/docs
```

## Configurare frontend

Instalare dependinte:

```bash
cd frontend/recipe-match
npm install
```

Pornire:

```bash
npm start
```

Sau direct pe platforma:

```bash
npm run android
npm run ios
npm run web
```

URL-ul backend-ului este in `constants/api.ts`:

```ts
const DEV_HOST = '172.20.10.4';
export const API_BASE_URL = __DEV__
  ? `http://${DEV_HOST}:8000`
  : 'https://your-production-api.com';
```

Pentru emulator Android, `localhost` nu inseamna masina host, ci emulatorul. De aceea fisierul foloseste IP-ul LAN al calculatorului. Pentru testare pe alt device, schimba `DEV_HOST` cu IP-ul masinii pe care ruleaza backend-ul.

## Comenzi utile

Frontend:

```bash
npm start
npm run android
npm run ios
npm run web
npm run lint
```

Backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Test manual pentru motorul Bayesian:

```bash
cd backend
.\venv\Scripts\Activate.ps1
python testari/test_app.py
```

Scripturile din `backend/testari` sunt pentru experimente, importuri, evaluari si variante anterioare ale algoritmului.

## Observatii importante

- Sesiunile de recomandare sunt tinute in memorie si au TTL de o ora. Daca backend-ul se restarteaza, sesiunile active se pierd.
- Motorul Bayesian persista sumarul sesiunii in `recommendation_sessions`, dar face asta fire-and-forget; daca persistenta esueaza, raspunsul catre utilizator nu este blocat.
- `recipe_interactions` poate stoca interactiuni precum `view`, `like`, `save`, `cook`, `skip`, dar frontend-ul curent foloseste in principal salvarea retetelor.
- In productie, CORS trebuie restrans in `main.py`; acum development permite `*`.
- `backend/.env` contine secrete si nu trebuie publicat.
- `frontend/recipe-match` are propriul folder `.git`, in timp ce radacina `recipe_match` nu pare configurata ca repository Git.

## Pe scurt

Recipe Match este o aplicatie full-stack de recomandari culinare. Frontend-ul ofera un UX mobil cu autentificare, tab-uri, swipe cards si retete salvate. Backend-ul gestioneaza autentificarea Supabase, citeste retete din baza de date, filtreaza dupa restrictii alimentare si ruleaza un motor Bayesian care alege intrebari adaptive si returneaza retete cu scor de potrivire.
