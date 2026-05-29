# Recipe Match

Recipe Match este o aplicatie full-stack pentru descoperire si recomandare de retete. Frontend-ul este o aplicatie mobila Expo/React Native, iar backend-ul este un API FastAPI care foloseste Supabase pentru autentificare, baza de date si operatii administrative.

Aplicatia combina mai multe mecanisme de recomandare:

- un motor Bayesian interactiv in tab-ul `Find`, care pune intrebari adaptive si actualizeaza probabilitatile retetelor dupa fiecare raspuns;
- reranking semantic cu embeddings pentru situatiile in care utilizatorul ofera un text de tip craving/preferinta;
- un feed `For You` hibrid, construit din interactiuni, raspunsuri istorice si trasee din Ingredient Explorer;
- un `Ingredient Explorer` bazat pe co-occurenta, PPMI si IDF pentru a sugera ingrediente compatibile;
- cautare de retete similare prin `pgvector`.

Scopul proiectului este sa ofere recomandari explicabile si personalizate, nu doar cautare dupa text. Retetele sunt filtrate dupa restrictii alimentare, apoi ordonate prin semnale comportamentale, statistice si semantice.

## Cuprins

- [Functionalitati](#functionalitati)
- [Stack tehnic](#stack-tehnic)
- [Structura proiectului](#structura-proiectului)
- [Arhitectura generala](#arhitectura-generala)
- [Date si modelare](#date-si-modelare)
- [Motorul Bayesian Find](#motorul-bayesian-find)
- [Reranking semantic](#reranking-semantic)
- [For You](#for-you)
- [Ingredient Explorer](#ingredient-explorer)
- [API backend](#api-backend)
- [Frontend](#frontend)
- [Migratii si tabele](#migratii-si-tabele)
- [Scripturi si evaluare](#scripturi-si-evaluare)
- [Configurare si rulare](#configurare-si-rulare)
- [Note de productie](#note-de-productie)

## Functionalitati

Aplicatia include:

- autentificare, inregistrare si profil utilizator prin Supabase Auth;
- profil alimentar: vegetarian, vegan, gluten-free, dairy-free;
- lista de ingrediente excluse manual;
- tab `Find` cu sesiune de recomandare prin intrebari;
- intrebari fixe si adaptive;
- rezultate cu `match_score` 0-100;
- tab `For You` cu recomandari personalizate;
- tab `Ingredient Explorer` pentru pornire de la un ingredient si extinderea unui chain;
- recomandari de retete pe baza ingredientelor selectate in Explorer;
- persistarea sesiunilor Explorer prin `explorer_sessions`;
- salvare retete;
- colectii pentru retete salvate;
- pagina de detaliu reteta;
- lista de cumparaturi;
- retete similare semantic;
- interactiuni de tip `view`, `like`, `save`, `cook`, `skip`;
- scripturi de precomputare embeddings si graf ingrediente;
- scripturi de evaluare pentru algoritmul Bayesian.

## Stack tehnic

### Frontend

- Expo SDK 54
- React 19
- React Native 0.81
- Expo Router
- Zustand pentru auth state
- AsyncStorage pentru token
- React Native Gesture Handler si Reanimated
- Expo Image
- TypeScript

### Backend

- FastAPI
- Pydantic
- pydantic-settings
- Supabase Python client
- Supabase Auth
- Supabase PostgREST
- SlowAPI pentru rate limiting
- NumPy pentru Bayesian inference
- sentence-transformers pentru embeddings

### Persistenta si cautare semantica

- Supabase PostgreSQL
- `auth.users`
- pgvector
- vector embeddings `vector(384)`
- RPC-uri SQL pentru similaritate cosine

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
        engine.py
        embeddings.py
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
      004_explorer_sessions.sql
      005_match_recipes_by_embedding.sql
    scripts/
      precompute_embeddings.py
      precompute_ingredient_graph.py
    testari/
      sensivity_analysis.py
      ...

  frontend/
    recipe-match/
      app/
        _layout.tsx
        (auth)/
          login.tsx
          register.tsx
        (tabs)/
          index.tsx
          find.tsx
          explorer.tsx
          saved.tsx
          profile.tsx
        recipe/
          [id].tsx
      components/
        RecipeCard.tsx
        LoadingSpinner.tsx
        explorer/
          GraphCanvas.tsx
          ExplorerResults.tsx
      services/
        api.ts
        explorerApi.ts
      hooks/
      store/
      constants/
```

## Arhitectura generala

Frontend-ul trimite requesturi catre backend cu JWT-ul Supabase in header:

```http
Authorization: Bearer <token>
```

Backend-ul valideaza tokenul in `middleware/auth.py`, extrage `sub` ca `user_id` si foloseste `get_supabase_admin()` pentru operatii cu service role acolo unde este nevoie.

La startup, `backend/app/main.py` face cateva incarcari importante:

1. Citeste toate retetele necesare motorului Bayesian din Supabase.
2. Le salveaza in `app.state.rec_recipes`.
3. Calculeaza ponderile intrebarilor cu `compute_feature_mi`.
4. Incalzeste modelul semantic printr-un embedding dummy.
5. Incalzeste cache-ul Ingredient Explorer.
6. Incarca IDF pentru toate ingredientele din `ingredient_stats`, paginat in loturi de 1000.

Sesiunile active de recomandare sunt tinute in memorie in backend, iar sumarul este persistat in `recommendation_sessions`.

## Date si modelare

Tabelul central este `recipes`. Pe langa campurile de afisare, retetele au feature-uri normalizate folosite de algoritmi:

- categorice: `meal_type`, `protein_type`, `cuisine`;
- dieta: `is_vegetarian`, `is_vegan`, `is_gluten_free`, `is_dairy_free`;
- gust si timp: `is_spicy`, `is_sweet`, `is_quick`;
- metoda: `needs_oven`, `needs_stovetop`, `is_no_cook`;
- ingrediente/baze: `has_pasta`, `has_rice`, `has_potato`, `has_tomato_base`, `has_cream_base`, `has_cheese`, `has_broth_base`, `has_mushroom`, `has_leafy_greens`, `has_beans_legumes`, `has_fruit`, `has_nuts`, `has_chocolate`, `has_asian_sauce`;
- ingrediente brute si curate: `ingredients`, `ingredients_clean`, `ingredients_clean_str`;
- embedding semantic: `embedding vector(384)`.

Filtrele alimentare sunt filtre dure. Daca utilizatorul este vegan, o reteta non-vegana nu intra in candidate pool. La fel pentru gluten-free, dairy-free si ingrediente excluse.

## Motorul Bayesian Find

Motorul este in `backend/app/recommender/engine.py`. Este folosit de endpointurile din `/recommendations`.

### Ideea principala

O sesiune porneste cu toate retetele candidate avand probabilitate uniforma. Pentru fiecare raspuns al utilizatorului, motorul calculeaza cat de compatibila este fiecare reteta cu raspunsul si actualizeaza distributia probabilitatilor.

Probabilitatile sunt tinute in log-space:

```text
log P(recipe | answers) += weight(question) * log likelihood(answer | recipe)
```

Asta evita underflow numeric cand sunt multe retete si multe update-uri.

### Parametri principali

In `engine.py`:

```python
MAX_QUESTIONS = 15
MIN_QUESTIONS_BEFORE_STOP = 4
POSTERIOR_TEMPERATURE = 1.25
P_CORRECT = 0.75
P_NOISE = 0.05
ENTROPY_STOP_THRESHOLD = math.log2(50)
```

`P_CORRECT` este probabilitatea atribuita unei retete care se potriveste cu raspunsul. `P_NOISE` este probabilitatea atribuita unei retete care nu se potriveste. Cu cat `P_NOISE` este mai mic, cu atat motorul penalizeaza mai dur nepotrivirile. Testele din `backend/testari/sensivity_analysis.py` arata ca valori prea agresive pot face targetul sa cada din top, pentru ca feature-urile sunt grosiere si uneori incomplete.

`POSTERIOR_TEMPERATURE` netezeste distributia. O temperatura mai mare face motorul mai putin sigur, deci poate cere mai multe intrebari, dar reduce riscul de convergenta prematura.

### Intrebari

Intrebarile fixe sunt puse primele:

1. `meal_type`
2. `cuisine`
3. `protein_type`

Intrebarile adaptive sunt booleene si acopera gust, timp, metoda de gatit si ingrediente:

- `is_spicy`
- `is_sweet`
- `is_quick`
- `needs_oven`
- `needs_stovetop`
- `is_no_cook`
- `has_pasta`
- `has_rice`
- `has_potato`
- `has_tomato_base`
- `has_cream_base`
- `has_cheese`
- `has_broth_base`
- `has_mushroom`
- `has_leafy_greens`
- `has_beans_legumes`
- `has_fruit`
- `has_nuts`
- `has_chocolate`
- `has_asian_sauce`

Exista excluderi logice intre metode de gatit. De exemplu, daca utilizatorul spune `yes` la `is_no_cook`, nu mai are sens sa fie intrebat `needs_oven` sau `needs_stovetop`.

### Likelihood

Pentru o intrebare categorica, daca reteta are exact valoarea aleasa, primeste `P_CORRECT`; altfel primeste `P_NOISE`.

Pentru multiselect, daca valoarea retetei este in lista selectata, primeste `P_CORRECT`; altfel `P_NOISE`.

Pentru boolean:

- raspuns `yes`: retetele care au feature-ul primesc `P_CORRECT`;
- raspuns `no`: retetele care nu au feature-ul primesc `P_CORRECT`;
- `skip`, `unknown`, `any`: likelihood 1.0, deci raspunsul nu schimba distributia.

### Ponderi prin Mutual Information

`compute_feature_mi()` estimeaza cat de informativ este fiecare feature adaptiv fata de tinta `(cuisine, protein_type)`. Rezultatul este normalizat in interval aproximativ `0.3 - 3.0`.

Aceste ponderi sunt folosite la actualizarea Bayesiană a posteriorului, controland cat de puternic modifica un raspuns distributia de probabilitate. Selectia intrebarii urmatoare ramane separata si este calculata prin expected entropy reduction pe starea curenta.

Intrebarile fixe primesc ponderi explicite:

```python
meal_type = 3.0
cuisine = 2.4
protein_type = 2.2
```

Acestea sunt importante pentru convergenta, deoarece primele raspunsuri trebuie sa restranga candidate pool-ul rapid.

### Alegerea urmatoarei intrebari

`select_next_question()` foloseste reducerea asteptata a entropiei. Pentru fiecare intrebare candidata, simuleaza raspunsurile posibile si estimeaza cat ar scadea entropia distributiei, fara a aplica ponderile globale MI in aceasta simulare.

Intrebarea aleasa este cea cu cel mai mare expected entropy reduction.

Daca utilizatorul a dat mai multe raspunsuri `no` sau `any`, motorul favorizeaza intrebari booleene cu prevalenta echilibrata in retetele relevante. Asta evita intrebari care ar separa prea putin candidate pool-ul.

### Conditia de oprire

Sesiunea nu se opreste in timpul intrebarilor fixe. Dupa numarul minim de intrebari, `should_stop()` verifica:

- entropia este sub prag;
- probabilitatea este suficient concentrata in top 10;
- topul este stabil intre ultimele update-uri;
- sau s-a atins `MAX_QUESTIONS`.

Sesiunea se opreste cand cel putin doua dintre conditiile principale sunt adevarate.

### Scorul afisat

`match_score` este un procent 0-100. In fluxul cu semantic rerank, scorul este clamp-uit defensiv ca sa nu depaseasca 100.

## Reranking semantic

Reranking-ul semantic este in `backend/app/recommender/semantic_rerank.py`.

Modelul folosit este:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding-urile au dimensiunea 384 si sunt normalizate. Similaritatea este produs scalar/cosine.

### Cand se activeaza

Semantic rerank se activeaza cand exista semnal semantic:

- `semantic_query`;
- `craving_text`;
- suficiente raspunsuri ca sa se construiasca un query textual util.

### Construirea query-ului

`build_preference_query()` transforma raspunsurile in text:

- cuisine -> `Italian recipe`, `Asian recipe`, etc.;
- meal type -> `for lunch or dinner`;
- protein -> `with chicken`;
- booleene pozitive -> `spicy`, `quick and easy`, `with cheese`;
- booleene negative -> `not spicy`, `without pasta`, etc.;
- semantic query este prepended.

### Combinarea scorurilor

Sunt doua moduri:

1. Blend semantic puternic:

```text
final = beta * bayes_norm + (1 - beta) * semantic_norm
```

2. Tie-breaker semantic:

```text
final = bayes_score + semantic_norm * TIE_BREAKER_POINTS
```

Rezultatul este clamp-uit intre 0 si 100 inainte de a ajunge la frontend.

### Warmup

Modelul Hugging Face se incarca lazy. Ca sa nu blocheze o sesiune la intrebarea 5-6, backend-ul face warmup la startup:

```python
encode_text("warmup recipe preferences")
```

Warning-ul despre `HF_TOKEN` nu este fatal; inseamna doar ca requesturile catre Hugging Face sunt neautentificate si pot avea rate limits mai mici.

## For You

`GET /foryou` este implementat in `backend/app/routers/foryou.py`.

Este un recomandator hibrid. Foloseste trei categorii de semnale:

1. interactiuni implicite;
2. raspunsuri din sesiuni completate;
3. trasee din Ingredient Explorer.

### Profil din interactiuni

`build_interaction_profile()` citeste:

- `recipe_interactions`
- `saved_recipes`

Interactiunile au ponderi:

```python
view = 0.5
like = 1.5
save = 2.0
cook = 3.0
```

Ponderea scade exponential in timp:

```text
weight = base_weight * exp(-0.02 * days_ago)
```

Profilul rezultat contine:

- distributii pentru `meal_type`, `protein_type`, `cuisine`;
- medii pentru feature-uri booleene;
- top ingrediente din retetele interactionate/salvate;
- `saved_ids`, pentru a evita recomandarea retetelor deja salvate.

### Profil din raspunsuri

`build_answers_profile()` citeste ultimele sesiuni completate din `recommendation_sessions`.

Pentru categorice, pastreaza valorile care apar in cel putin aproximativ 40% din sesiunile recente. Pentru booleene, decide `yes` sau `no` daca semnalul este suficient de frecvent.

### Profil din Explorer

`build_explorer_profile()` citeste `explorer_sessions`.

Fiecare sesiune Explorer este un singur rand, identificat prin `session_id` generat in frontend. Backend-ul face upsert pe acelasi ID, deci un chain incremental nu este numarat de mai multe ori.

Exemplu:

```text
["spaghetti", "parmesan cheese"]
["spaghetti", "parmesan cheese", "bacon"]
```

devine un singur rand actualizat, nu doua sesiuni separate.

For You numara fiecare ingredient o singura data per sesiune si pastreaza ingredientele care apar in cel putin 30% din sesiunile recente.

### Profil semantic

`build_semantic_profile()` construieste textul pentru embedding:

1. ingrediente frecvente din Explorer;
2. ingrediente din interactiuni;
3. top valori categorice;
4. feature-uri booleene pozitive.

Textul este embed-uit si trimis catre RPC-ul:

```sql
match_recipes_by_embedding(query_embedding vector(384), match_count int)
```

### Scor final

Pentru fiecare candidat:

```text
score =
  w_interactions * interaction_score
  + w_answers * answers_score
  + w_semantic * semantic_score
```

Ponderile depind de cate semnale exista:

- fara interactiuni, dar cu sesiuni: answers 0.60, semantic 0.40;
- fara sesiuni, dar cu interactiuni: interactions 0.55, semantic 0.45;
- cu ambele: ponderi cresc gradual cu numarul de interactiuni si sesiuni, semantic ramane minim 0.15;
- daca exista doar Explorer, semantic devine 1.0.

Candidatii provin din doua pool-uri:

- top semantic din RPC;
- top profil dupa matching pe feature-uri.

La final, `_diverse_top()` limiteaza repetitia de cuisine si meal type, ca feed-ul sa fie mai variat.

### Cold start

Daca utilizatorul nu are interactiuni, sesiuni si nici Explorer, For You intoarce retete populare. Popularitatea este estimata din `recipe_interactions` (`save`, `cook`) si `saved_recipes`.

## Ingredient Explorer

Ingredient Explorer este implementat in `backend/app/routers/explorer.py` si in frontend in `app/(tabs)/explorer.tsx`.

Scopul lui este sa ajute utilizatorul sa construiasca un chain de ingrediente compatibile si apoi sa vada retete care contin acel chain.

### Date folosite

Explorer foloseste:

- `recipes.ingredients_clean`;
- `ingredient_stats`;
- `ingredient_graph`;
- cache in memorie pentru randurile retetelor;
- IDF pentru ingrediente.

La startup:

- `warm_explorer_cache()` incarca retetele si ingredientele curatate;
- `warm_ingredient_idf()` incarca toate randurile din `ingredient_stats`, paginat.

### Start

`POST /explorer/start` primeste un ingredient. Backend-ul:

1. normalizeaza ingredientul;
2. verifica daca exista in `ingredient_stats`;
3. gaseste retetele care il contin;
4. calculeaza ingrediente candidate din acele retete;
5. filtreaza ingrediente de pantry;
6. combina frecventa, IDF si PPMI;
7. returneaza top 5 sugestii.

### Expand

`POST /explorer/expand` primeste:

```json
{
  "selected_ingredients": ["spaghetti", "parmesan cheese"],
  "session_id": "...",
  "finalize": false
}
```

Backend-ul cauta retete care contin toate ingredientele selectate. Sugestiile noi sunt calculate cu:

```text
frequency_score = count_in_matching_recipes / number_of_matching_recipes
tfidf_score = frequency_score * idf
shifted_ppmi = max(avg_ppmi - log(k), 0)
final_score = 0.7 * tfidf_score + 0.3 * shifted_ppmi
```

Sugestiile sunt sortate descrescator dupa `final_score`.

### Recommend

`POST /explorer/recommend` intoarce retete care contin ingredientele selectate, sortate dupa Jaccard:

```text
jaccard = |selected ingredients intersect recipe ingredients| / |selected union recipe ingredients|
```

### Persistarea sesiunii Explorer

Frontend-ul genereaza un UUID cand incepe o sesiune noua de Explorer. La fiecare expand sau recommend trimite acelasi `session_id`.

Backend-ul face upsert in `explorer_sessions`:

- `id`: session id;
- `user_id`;
- `chain`;
- `updated_at`;
- `finalized_at` cand userul apasa Find Recipes.

Acest tabel nu influenteaza sugestiile Explorer. El este citit doar de For You pentru personalizare.

## API backend

Toate raspunsurile sunt invelite in:

```json
{
  "data": {},
  "error": null
}
```

### Auth

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| POST | `/auth/register` | Creeaza utilizator Supabase si metadata alimentara |
| POST | `/auth/login` | Autentifica si returneaza JWT |
| POST | `/auth/logout` | Logout |
| GET | `/auth/me` | Profil curent |
| PATCH | `/auth/me` | Actualizare profil alimentar |

### Recipes

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| GET | `/recipes/ingredients` | Sugestii de ingrediente |
| GET | `/recipes/{recipe_id}` | Detalii reteta |
| GET | `/recipes/{recipe_id}/shopping-list` | Lista de cumparaturi |
| GET | `/recipes/{recipe_id}/similar` | Retete similare prin pgvector |

### Recommendations

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| POST | `/recommendations/session/start` | Porneste sesiune Bayesian |
| POST | `/recommendations/session/{session_id}/answer` | Trimite raspuns |
| GET | `/recommendations/session/{session_id}/results` | Rezultate sesiune |
| POST | `/recommendations/interaction` | Salveaza interactiune |

### For You

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| GET | `/foryou` | Feed personalizat hibrid |

### Explorer

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| GET | `/explorer/search?q=` | Cauta ingrediente |
| POST | `/explorer/start` | Porneste de la un ingredient |
| POST | `/explorer/expand` | Extinde chain-ul |
| POST | `/explorer/recommend` | Recomanda retete pentru chain |

### Saved si Collections

| Metoda | Endpoint | Descriere |
| --- | --- | --- |
| GET | `/saved` | Retete salvate |
| POST | `/saved` | Salveaza reteta |
| DELETE | `/saved/{recipe_id}` | Sterge reteta salvata |
| GET | `/saved/collections/{collection_id}` | Retete din colectie |
| GET | `/collections` | Listeaza colectii |
| POST | `/collections` | Creeaza colectie |
| DELETE | `/collections/{collection_id}` | Sterge colectie |

## Frontend

### Navigatie

Expo Router imparte aplicatia in:

- `(auth)`: login si register;
- `(tabs)`: zona autentificata;
- `recipe/[id]`: detalii reteta.

Tab-urile principale:

- `index.tsx`: For You;
- `find.tsx`: sesiune Bayesian;
- `explorer.tsx`: Ingredient Explorer;
- `saved.tsx`: salvate si colectii;
- `profile.tsx`: profil utilizator.

### Auth store

`store/authStore.ts` tine:

- tokenul;
- profilul utilizatorului;
- starea de initializare;
- actiuni login/register/logout/update.

Tokenul este persistat in AsyncStorage.

### Clienti HTTP

`services/api.ts` contine clientul principal pentru auth, recipes, saved, collections si recommendations.

`services/explorerApi.ts` contine clientul Explorer si generatorul local de session id:

```ts
createExplorerSessionId()
```

### Find UI

`app/(tabs)/find.tsx` afiseaza:

- progresul intrebarilor;
- intrebari categorice/multiselect ca optiuni;
- intrebari booleene ca swipe/actiuni;
- rezultate cu badge `match_score`.

### Explorer UI

`app/(tabs)/explorer.tsx` gestioneaza:

- cautarea ingredientului initial;
- session id generat la start;
- chain-ul selectat;
- sugestiile din backend;
- butonul Find Recipes;
- ecranul de rezultate Explorer.

`GraphCanvas` afiseaza nodul central, ingredientele anterioare si sugestiile.

## Migratii si tabele

### `001_recipe_features.sql`

Adauga feature-uri booleene pe `recipes` si creeaza:

- `recommendation_sessions`;
- `recipe_interactions`.

`recommendation_sessions` pastreaza sumarul sesiunilor:

- `id`;
- `user_id`;
- `answers`;
- `question_order`;
- `questions_asked`;
- `entropy_final`;
- `top_recipe_ids`;
- `completed_at`.

`recipe_interactions` pastreaza:

- `user_id`;
- `recipe_id`;
- `interaction_type`;
- `weight`;
- `created_at`.

### `002_recipe_embeddings.sql`

Activeaza pgvector si adauga:

- `recipes.embedding vector(384)`;
- index ivfflat pentru cautare vectoriala;
- RPC pentru retete similare.

### `003_ingredient_explorer.sql`

Creeaza:

- `ingredient_graph`;
- `ingredient_stats`.

`ingredient_graph` contine perechi de ingrediente si scoruri PPMI.

`ingredient_stats` contine frecventa fiecarui ingredient.

### `004_explorer_sessions.sql`

Creeaza:

```sql
public.explorer_sessions (
  id uuid primary key,
  user_id uuid references auth.users(id),
  chain text[] not null,
  started_at timestamp,
  updated_at timestamp,
  finalized_at timestamp
)
```

Este folosit pentru personalizarea For You pe baza explorarii ingredientelor.

### `005_match_recipes_by_embedding.sql`

Creeaza RPC:

```sql
match_recipes_by_embedding(
  query_embedding vector(384),
  match_count int default 100
)
```

Returneaza `id` si `similarity` pentru cele mai apropiate retete dupa embedding.

## Scripturi si evaluare

### Embeddings

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python scripts\precompute_embeddings.py
```

Calculeaza embedding pentru fiecare reteta si salveaza in `recipes.embedding`.

### Graf ingrediente

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python scripts\precompute_ingredient_graph.py
```

Construieste `ingredient_graph` si `ingredient_stats` din `ingredients_clean`.

### Sensitivity analysis

`backend/testari/sensivity_analysis.py` testeaza combinatii de `P_CORRECT` si `P_NOISE`.

Metrici:

- Avg Q;
- Med Q;
- HR@10;
- HR@20;
- RankFound;
- cate sesiuni se opresc prin `should_stop` vs `max_questions`;
- scenarii cu raspunsuri perfecte si raspunsuri noisy.

Rezultatele recente au aratat ca `P_CORRECT=0.90, P_NOISE=0.05` este prea overconfident pentru datele curente. Zona `P_CORRECT=0.75` cu `P_NOISE` intre `0.07` si `0.10` este mai robusta, cu tradeoff intre acuratete si numar de intrebari.

## Configurare si rulare

### Backend `.env`

Fisier: `backend/.env`

```env
SUPABASE_URL=
SUPABASE_KEY=
SUPABASE_SERVICE_KEY=
SECRET_KEY=
ENVIRONMENT=development
```

`SUPABASE_SERVICE_KEY` nu trebuie expus in frontend.

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

Swagger in development:

```text
http://localhost:8000/docs
```

### Pornire frontend

```powershell
cd frontend\recipe-match
npm install
npm start
```

Comenzi:

```powershell
npm run android
npm run ios
npm run web
npm run lint
```

URL-ul backend-ului este in:

```text
frontend/recipe-match/constants/api.ts
```

Pe device sau emulator, `localhost` nu indica intotdeauna masina host. De aceea, in development se foloseste IP-ul LAN al calculatorului.

## Note de productie

- `backend/.env` contine secrete si nu trebuie commit-uit.
- `SUPABASE_SERVICE_KEY` ramane doar pe backend.
- CORS trebuie restrans in productie.
- Sesiunile active din `/recommendations` sunt in memorie si se pierd la restart.
- `recommendation_sessions` pastreaza sumarul pentru profilare si analiza, nu starea completa runtime.
- Modelul Hugging Face poate descarca/incarca greutati la prima folosire; warmup-ul de startup reduce blocajele in requesturile userului.
- Pentru rate limits mai bune la Hugging Face, se poate seta `HF_TOKEN`.
- `explorer_sessions` nu afecteaza sugestiile Explorer; afecteaza doar personalizarea For You.
- Daca numarul de retete creste mult, incarcarea tuturor retetelor in memorie ar trebui inlocuita cu cache persistent, job-uri batch sau cautare indexata mai agresiv.
- Algoritmii sunt calibrabili: `P_CORRECT`, `P_NOISE`, ponderile intrebarilor, pragurile de oprire si mixul semantic pot fi ajustate pe baza scripturilor din `backend/testari`.
