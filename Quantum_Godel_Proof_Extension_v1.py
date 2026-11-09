"""
Quantum_Godel_Proof_Extension_V1.py

Python 3.11
Qiskit 1.4.4

POSEBNOST MODELA
---------------
Model uci raspodelu nad dopustivim tumacenjima delimicnih teorija.

Svaki buduci broj u pojedinacnoj teoriji moze biti:
    izvedeno ukljucen;
    izvedeno iskljucen;
    nerazresen u zadatom budzetu dokazivanja.

Nerazreseno se ne tretira kao netacno.

Tvrdnje i pravila imaju jednoznacne aritmeticke kodove.
Izvodi imaju nezavisno proverljive sertifikate.

Ovo je konacna prediktivna adaptacija.
Ogranicena pretraga dokaza ne predstavlja dokaz
Godelovske nedokazivosti.

FORMALNI SLOJ
-------------
Istorijske cinjenice:
    prisustvo ili odsustvo broja u prethodnom kolu;
    prisustvo ili odsustvo u pretprethodnom kolu,
    kada ono postoji.

Naucene hipoteze:
    istorijski uslov -> pomocna tvrdnja;
    izabrano prosirenje: pomocna tvrdnja -> buduci literal;
    buduci literal -> drugi buduci literal.

Statisticka povezanost sluzi za konstrukciju hipoteza.
Izvod dokazuje posledicu unutar hipoteze,
a ne istinitost buduceg izvlacenja.

DOKAZNI BUDZET
--------------
Svaki korak izvoda povecava dubinu za jedan.
Pravila u istom koraku ne koriste novodobijene zakljucke.

Proverava se:
    kod tvrdnje i pravila;
    raspolozivost premise;
    pripadnost pravila teoriji;
    dubina izvoda;
    saglasnost pozitivnih i negativnih zakljucaka.

Provera saglasnosti odnosi se na konacnu propozicionu bazu.

QISKIT
------
Sedam pomocnih teorijskih kubita.
Jedan pomocni kubit za logicki orakl.

Brojna raspodela koristi sest kubita.
Sedam izlaznih registara predstavljaju sedam kopija tog stanja.

RY rotacije zavise od istorije.
CNOT lanci povezuju kubite.

Qiskit orakl oznacava:
    saglasna prosirenja;
    prosirenja koja dopustaju konkretnu kombinaciju.

Gust vektor svih izlaznih registara se ne pravi.
Koristi se tacna faktorisana i simbolicka simulacija.

RASPodela
---------
Za teoriju T i skup S:

    masa(T,S) =
        q(T) * product_{v in S} p(v)
        * indicator(T je saglasna)
        * indicator(S zadovoljava izvedene literale T)

Posle zajednicke normalizacije:

    P(S) =
        product_{v in S} p(v)
        * sum_{T koji dopustaju S} q(T)
        / Z

Z obuhvata sve legalne skupove od sedam brojeva.

Logicki atom predstavlja sest brojeva izabranih prema
uslovnoj informacionoj povezanosti u treningu.
Svi ostali brojevi ostaju u punoj prediktivnoj raspodeli.

Za tacnu normalizaciju sabiraju se svi logicki obrasci.
Preostali izbori racunaju se elementarnim simetricnim polinomima.

Globalni MAP:
    za svaki logicki obrazac odredjuje se najbolji slobodni ostatak;
    zatim se porede kompletne kombinacije svih obrazaca.

UCENJE
------
NLL celih kombinacija i analiticki gradijent.
L-BFGS-B optimizuje parametre kvantnih rotacija.

Pravila, atomizacija i skaliranje uce se samo na treningu.
Validacija bira dokazni budzet i regularizaciju.
Test je odvojen.
Zavrsno ucenje koristi sve istorijske prelaze.

Nema klasifikatora, slucajnih kombinacija, dodatog suma,
zaokruzivanja predikcija ili kozmetickih filtera.

DIAGNOSTICS=False daje kratke ispise.
"""

import csv
import hashlib
import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from scipy.special import logsumexp, rel_entr


random.seed(39)
np.random.seed(39)


FILES = (
    Path("/data/loto7_4716_k89.csv"),
    Path("/data/loto7_4716_k89_loto_2980.csv"),
    Path("/data/loto7_4716_k89_loto_plus_1736.csv"),
)

DIAGNOSTICS = False

N = 39
K = 7
ATOM_COUNT = 6

THEORY_QUBITS = 7
THEORY_DIM = 1 << THEORY_QUBITS

NUMBER_QUBITS = 6
NUMBER_DIM = 1 << NUMBER_QUBITS

GATE_COUNT = THEORY_QUBITS + NUMBER_QUBITS

PROOF_BUDGETS = (2, 4)
REGULARIZATIONS = (0.01, 0.1)

TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

BATCH_SIZE = 64
RESTARTS = 2
MAX_ITERATIONS = 150
GRADIENT_TOLERANCE = 1e-5

PAST_ONE = 0
PAST_TWO = 1
BRIDGE = 2
FUTURE = 3

PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19,
    23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89,
    97, 101, 103, 107, 109, 113, 127, 131,
)


class DistributionError(ArithmeticError):
    pass


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def detail(*values):
    if DIAGNOSTICS:
        print(*values, flush=True)


def godel_encode(tokens):
    tokens = tuple(int(token) for token in tokens)

    require(
        len(tokens) <= len(PRIMES)
        and all(token >= 0 for token in tokens),
        "Neispravan niz simbola.",
    )

    code = 1
    for prime, token in zip(PRIMES, tokens):
        code *= pow(prime, token + 1)
    return code


def godel_decode(code):
    code = int(code)
    require(code >= 1, "Neispravan aritmeticki kod.")

    tokens = []

    for prime in PRIMES:
        if code == 1:
            break

        exponent = 0
        while code % prime == 0:
            exponent += 1
            code //= prime

        require(exponent > 0, "Kod sadrzi prazninu u nizu simbola.")
        tokens.append(exponent - 1)

    require(code == 1, "Kod prelazi dozvoljeni alfabet.")
    return tuple(tokens)


@dataclass(frozen=True)
class Atom:
    kind: int
    index: int
    truth: int

    def tokens(self):
        return (0, int(self.kind), int(self.index), int(self.truth))

    @property
    def code(self):
        return godel_encode(self.tokens())


@dataclass(frozen=True)
class Rule:
    premise: Atom
    conclusion: Atom

    def tokens(self):
        return (1,) + self.premise.tokens() + self.conclusion.tokens()

    @property
    def code(self):
        return godel_encode(self.tokens())


def register_description(qubits):
    dimension = 1 << qubits
    basis = np.arange(dimension, dtype=np.int64)
    bits = (
        basis[:, None]
        >> np.arange(qubits, dtype=np.int64)[None, :]
    ) & 1

    permutation = basis.copy()

    for control in range(qubits - 1):
        permutation ^= (
            ((permutation >> control) & 1) << (control + 1)
        )

    require(
        len(np.unique(permutation)) == dimension,
        "CNOT niz nije permutacija.",
    )

    return {
        "bits": bits,
        "permutation": permutation,
        "inverse": np.argsort(permutation),
    }


THEORY_REGISTER = register_description(THEORY_QUBITS)
NUMBER_REGISTER = register_description(NUMBER_QUBITS)


def load_csv(path):
    raw = path.read_bytes()
    rows = []

    for line_number, row in enumerate(
        csv.reader(raw.decode("utf-8-sig").splitlines()),
        start=1,
    ):
        if not row or all(not cell.strip() for cell in row):
            continue

        require(
            len(row) == K,
            f"{path.name}, red {line_number}: potrebno je sedam kolona.",
        )

        try:
            values = [int(cell.strip()) for cell in row]
        except ValueError as exc:
            raise RuntimeError(
                f"{path.name}, red {line_number}: necelobrojna vrednost."
            ) from exc

        require(
            len(set(values)) == K
            and all(1 <= value <= N for value in values)
            and values == sorted(values),
            f"{path.name}, red {line_number}: nelegalna kombinacija.",
        )
        rows.append(values)

    require(len(rows) >= 5, f"{path.name}: nedovoljno kola.")

    return {
        "path": path,
        "data": np.asarray(rows, dtype=np.int64) - 1,
        "digest": hashlib.sha256(raw).hexdigest(),
    }


def history_patterns(data, times):
    times = np.asarray(times, dtype=np.int64)

    require(
        times.ndim == 1
        and len(times) > 0
        and np.all((times >= 1) & (times <= len(data))),
        "Neispravna istorijska vremena.",
    )

    patterns = np.zeros((len(times), 2 * N + 1), dtype=bool)
    rows = np.arange(len(times))[:, None]

    patterns[rows, data[times - 1]] = True

    available = np.flatnonzero(times >= 2)

    if len(available):
        patterns[
            available[:, None],
            N + data[times[available] - 2],
        ] = True

    patterns[:, -1] = times >= 2
    return patterns


def target_presence(targets):
    result = np.zeros((len(targets), N), dtype=bool)
    result[np.arange(len(targets))[:, None], targets] = True
    return result


def fit_transform(patterns, training_rows):
    require(
        1 <= training_rows <= len(patterns),
        "Neispravna granica skaliranja.",
    )

    training = patterns[:training_rows].astype(float)
    mean = training.mean(axis=0)
    scale = training.std(axis=0)

    return {
        "mean": mean,
        "scale": scale,
        "keep": scale > 1e-12,
    }


def transform_features(patterns, transform):
    keep = transform["keep"]
    features = np.ones(
        (len(patterns), 1 + int(np.sum(keep))),
        dtype=np.float64,
    )

    if np.any(keep):
        features[:, 1:] = (
            patterns[:, keep].astype(float)
            - transform["mean"][keep]
        ) / transform["scale"][keep]

    require(np.isfinite(features).all(), "Nekonacne karakteristike.")
    return features


def conditional_scores(conditions, outcomes, minimum_support):
    conditions = np.asarray(conditions, dtype=bool)
    outcomes = np.asarray(outcomes, dtype=bool)

    counts = conditions.sum(axis=0).astype(float)
    joint = conditions.astype(float).T @ outcomes.astype(float)
    baseline = outcomes.mean(axis=0)

    conditional = np.zeros_like(joint)

    np.divide(
        joint,
        counts[:, None],
        out=conditional,
        where=counts[:, None] > 0.0,
    )

    scores = np.zeros_like(conditional)
    valid_rows = np.flatnonzero(counts >= minimum_support)
    valid_columns = np.flatnonzero(
        (baseline > 0.0) & (baseline < 1.0)
    )

    if len(valid_rows) and len(valid_columns):
        probabilities = conditional[
            np.ix_(valid_rows, valid_columns)
        ]
        reference = baseline[valid_columns][None, :]

        divergence = (
            rel_entr(probabilities, reference)
            + rel_entr(1.0 - probabilities, 1.0 - reference)
        )

        scores[np.ix_(valid_rows, valid_columns)] = (
            counts[valid_rows, None]
            / len(outcomes)
            * np.maximum(divergence, 0.0)
        )

    return scores, conditional, baseline


def assemble_graph(atoms, conditions, links):
    atoms = np.asarray(atoms, dtype=np.int64)
    count = len(atoms)

    require(
        count > 0
        and len(np.unique(atoms)) == count
        and np.all((atoms >= 0) & (atoms < N))
        and len(conditions) == 2 * count,
        "Neispravna formalna baza.",
    )

    roots = []
    extensions = []

    for index in range(2 * count):
        atom_index = index // 2
        truth = index % 2
        bridge = Atom(BRIDGE, index, 1)

        condition = conditions[index]

        if condition is None:
            roots.append(None)
        else:
            feature, condition_truth = condition
            roots.append(Rule(
                Atom(feature // N, feature % N, condition_truth),
                bridge,
            ))

        extensions.append(Rule(
            bridge,
            Atom(FUTURE, int(atoms[atom_index]), truth),
        ))

    link_rules = [
        Rule(
            Atom(FUTURE, int(atoms[source]), source_truth),
            Atom(FUTURE, int(atoms[target]), target_truth),
        )
        for source, source_truth, target, target_truth in links
    ]

    masks = [0]

    for size in (1, 2):
        for selected in itertools.combinations(range(2 * count), size):
            masks.append(sum(1 << index for index in selected))

    masks = np.asarray(masks, dtype=np.int64)

    require(
        len(masks) <= THEORY_DIM,
        "Prosirenja ne staju u teorijski registar.",
    )

    return {
        "atoms": atoms,
        "conditions": tuple(conditions),
        "roots": tuple(roots),
        "extensions": tuple(extensions),
        "links": tuple(links),
        "link_rules": tuple(link_rules),
        "masks": masks,
        "selected": (
            (masks[:, None] >> np.arange(2 * count)[None, :]) & 1
        ).astype(bool),
    }


def learn_graph(patterns, targets):
    require(
        len(patterns) == len(targets) and len(targets) > 0,
        "Neispravan trening formalne baze.",
    )

    outcomes = target_presence(targets)
    condition_columns = []
    condition_descriptions = []

    for feature in range(2 * N):
        for truth in (0, 1):
            condition = patterns[:, feature] == bool(truth)

            if feature >= N:
                condition = condition & patterns[:, -1]

            condition_columns.append(condition)
            condition_descriptions.append((feature, truth))

    conditions = np.stack(condition_columns, axis=1)
    minimum_support = max(8, int(0.02 * len(targets)))

    scores, probabilities, baseline = conditional_scores(
        conditions, outcomes, minimum_support
    )

    head_scores = scores.max(axis=0)
    atoms = np.lexsort((np.arange(N), -head_scores))[:ATOM_COUNT]

    selected_conditions = []

    for atom in atoms:
        for desired_truth in (0, 1):
            direction = (
                probabilities[:, atom] > baseline[atom]
                if desired_truth == 1
                else probabilities[:, atom] < baseline[atom]
            )

            candidate_scores = np.where(
                direction, scores[:, atom], -np.inf
            )
            selected = int(np.argmax(candidate_scores))

            if (
                not np.isfinite(candidate_scores[selected])
                or candidate_scores[selected] <= 1e-12
            ):
                selected_conditions.append(None)
            else:
                selected_conditions.append(
                    condition_descriptions[selected]
                )

    selected_outcomes = outcomes[:, atoms]
    link_conditions = np.stack([
        selected_outcomes[:, index] == bool(truth)
        for index in range(len(atoms))
        for truth in (0, 1)
    ], axis=1)

    link_scores, link_probabilities, link_baseline = conditional_scores(
        link_conditions, selected_outcomes, minimum_support
    )

    links = []

    for source in range(len(atoms)):
        for truth in (0, 1):
            row = 2 * source + truth
            candidate_scores = link_scores[row].copy()
            candidate_scores[source] = -np.inf
            target = int(np.argmax(candidate_scores))

            if (
                not np.isfinite(candidate_scores[target])
                or candidate_scores[target] <= 1e-12
            ):
                continue

            target_truth = int(
                link_probabilities[row, target] > link_baseline[target]
            )
            links.append((source, truth, target, target_truth))

    return assemble_graph(atoms, selected_conditions, links)


def compile_closure(graph, patterns, budget):
    require(budget >= 1, "Neispravan dokazni budzet.")

    batch = len(patterns)
    count = len(graph["atoms"])
    theory_count = len(graph["masks"])

    bridge_active = np.zeros((batch, 2 * count), dtype=bool)

    for index, condition in enumerate(graph["conditions"]):
        if condition is None:
            continue

        feature, truth = condition
        active = patterns[:, feature] == bool(truth)

        if feature >= N:
            active &= patterns[:, -1]

        bridge_active[:, index] = active

    positive = np.zeros((batch, theory_count), dtype=np.int64)
    negative = np.zeros_like(positive)

    if budget >= 2:
        fired = (
            bridge_active[:, None, :]
            & graph["selected"][None, :, :]
        )

        for index in range(2 * count):
            bit = 1 << (index // 2)

            if index % 2:
                positive |= np.where(fired[:, :, index], bit, 0)
            else:
                negative |= np.where(fired[:, :, index], bit, 0)

    for _ in range(3, budget + 1):
        next_positive = positive.copy()
        next_negative = negative.copy()

        for source, source_truth, target, target_truth in graph["links"]:
            source_mask = positive if source_truth else negative
            active = ((source_mask >> source) & 1) != 0
            addition = np.where(active, 1 << target, 0)

            if target_truth:
                next_positive |= addition
            else:
                next_negative |= addition

        unchanged = (
            np.array_equal(next_positive, positive)
            and np.array_equal(next_negative, negative)
        )
        positive, negative = next_positive, next_negative

        if unchanged:
            break

    return {"positive": positive, "negative": negative}


def input_facts(pattern):
    facts = set()

    for number in range(N):
        facts.add(Atom(PAST_ONE, number, int(pattern[number])))

    if pattern[-1]:
        for number in range(N):
            facts.add(Atom(
                PAST_TWO, number, int(pattern[N + number])
            ))

    return facts


def rules_for_theory(graph, theory_index):
    mask = int(graph["masks"][theory_index])
    rules = [rule for rule in graph["roots"] if rule is not None]

    rules.extend(
        rule
        for index, rule in enumerate(graph["extensions"])
        if (mask >> index) & 1
    )
    rules.extend(graph["link_rules"])
    return rules


def derive_certificate(graph, pattern, theory_index, budget):
    facts = input_facts(pattern)
    known = {}
    records = []

    for atom in sorted(facts, key=lambda value: value.code):
        known[atom] = 0
        records.append((atom, 0, 0, None))

    rules = rules_for_theory(graph, theory_index)

    for _ in range(budget):
        previous = set(known)
        added = False

        for rule in rules:
            if rule.premise not in previous:
                continue
            if rule.conclusion in known:
                continue

            depth = known[rule.premise] + 1

            if depth > budget:
                continue

            known[rule.conclusion] = depth
            records.append((
                rule.conclusion,
                depth,
                rule.code,
                rule.premise,
            ))
            added = True

        if not added:
            break

    return tuple(records)


def verify_certificate(graph, pattern, theory_index, budget, records):
    facts = input_facts(pattern)
    allowed = {
        rule.code: rule
        for rule in rules_for_theory(graph, theory_index)
    }
    known = {}

    for atom, depth, rule_code, premise in records:
        require(
            godel_decode(atom.code) == atom.tokens()
            and atom not in known,
            "Neispravan kod ili ponovljen korak sertifikata.",
        )

        if rule_code == 0:
            require(
                atom in facts and depth == 0 and premise is None,
                "Neispravna pocetna cinjenica.",
            )
        else:
            require(rule_code in allowed, "Pravilo nije u teoriji.")
            rule = allowed[rule_code]

            require(
                godel_decode(rule_code) == rule.tokens()
                and rule.premise == premise
                and rule.conclusion == atom
                and premise in known
                and depth == known[premise] + 1
                and 1 <= depth <= budget,
                "Neispravan izvod u sertifikatu.",
            )

        known[atom] = depth

    positive = 0
    negative = 0
    positions = {
        int(number): index
        for index, number in enumerate(graph["atoms"])
    }

    for atom in known:
        if atom.kind != FUTURE:
            continue

        require(atom.index in positions, "Zakljucak je van formalne baze.")
        bit = 1 << positions[atom.index]

        if atom.truth:
            positive |= bit
        else:
            negative |= bit

    return positive, negative


def log_amplitudes(amplitudes):
    magnitude = np.abs(amplitudes)
    result = np.full(magnitude.shape, -np.inf)
    positive = magnitude > 0.0
    result[positive] = 2.0 * np.log(magnitude[positive])
    return result


def register_logs(amplitudes, description):
    bits = description["bits"]

    raw_logs = np.sum(np.where(
        bits[None, :, :] == 1,
        log_amplitudes(amplitudes)[:, None, :, 1],
        log_amplitudes(amplitudes)[:, None, :, 0],
    ), axis=2)

    logs = raw_logs[:, description["inverse"]]

    require(
        np.allclose(
            logsumexp(logs, axis=1), 0.0,
            atol=2e-9, rtol=0.0,
        ),
        "Kvantni registar nije normalizovan.",
    )
    return logs


def free_polynomials(log_weights, maximum_k, need_marginals):
    batch, states = log_weights.shape

    forward = np.full(
        (batch, states + 1, maximum_k + 1), -np.inf
    )
    forward[:, 0, 0] = 0.0

    for value in range(states):
        forward[:, value + 1, 0] = 0.0
        forward[:, value + 1, 1:] = np.logaddexp(
            forward[:, value, 1:],
            log_weights[:, value, None] + forward[:, value, :-1],
        )

    log_e = forward[:, -1]

    if not need_marginals:
        return log_e, None

    backward = np.full_like(forward, -np.inf)
    backward[:, states, 0] = 0.0

    for value in range(states - 1, -1, -1):
        backward[:, value, 0] = 0.0
        backward[:, value, 1:] = np.logaddexp(
            backward[:, value + 1, 1:],
            log_weights[:, value, None] + backward[:, value + 1, :-1],
        )

    inclusion = np.zeros(
        (batch, maximum_k + 1, states),
        dtype=np.float64,
    )

    for cardinality in range(1, maximum_k + 1):
        valid = np.isfinite(log_e[:, cardinality])

        for value in range(states):
            numerator = (
                log_weights[:, value]
                + logsumexp(
                    forward[:, value, :cardinality]
                    + backward[
                        :, value + 1, :cardinality
                    ][:, ::-1],
                    axis=1,
                )
            )
            ratio = np.full(batch, -np.inf)
            np.subtract(
                numerator,
                log_e[:, cardinality],
                out=ratio,
                where=valid,
            )
            inclusion[:, cardinality, value] = np.exp(ratio)

        require(
            np.allclose(
                inclusion[valid, cardinality].sum(axis=1),
                float(cardinality),
                atol=2e-8, rtol=0.0,
            ),
            "Slobodni izbor nije normalizovan.",
        )

    return log_e, inclusion


def logical_distribution(
    log_q, log_p, positive, negative, atoms,
    cardinality=K, need_marginals=True,
):
    batch, states = log_p.shape
    theory_count = log_q.shape[1]
    atom_count = len(atoms)

    require(
        positive.shape == negative.shape == (batch, theory_count)
        and log_q.shape[0] == batch
        and 0 < atom_count <= states
        and 1 <= cardinality <= states
        and not np.isnan(log_q).any()
        and not np.isnan(log_p).any()
        and not np.isposinf(log_q).any()
        and not np.isposinf(log_p).any()
        and np.all(positive >= 0)
        and np.all(negative >= 0)
        and np.all(positive < (1 << atom_count))
        and np.all(negative < (1 << atom_count)),
        "Neispravan logicki model.",
    )

    require(
        np.all(logsumexp(log_q, axis=1) <= 2e-9)
        and np.all(logsumexp(log_p, axis=1) <= 2e-9),
        "Ulazne kvantne mase su vece od jedan.",
    )

    atoms = np.asarray(atoms, dtype=np.int64)
    free = np.array(
        [value for value in range(states) if value not in set(atoms)],
        dtype=np.int64,
    )

    assignments = np.arange(1 << atom_count, dtype=np.int64)
    assignment_bits = (
        assignments[:, None] >> np.arange(atom_count)[None, :]
    ) & 1
    selected_count = assignment_bits.sum(axis=1)
    free_count = cardinality - selected_count
    valid_pattern = (free_count >= 0) & (free_count <= len(free))

    log_e, free_inclusion = free_polynomials(
        log_p[:, free], cardinality, need_marginals
    )

    atom_scores = np.sum(np.where(
        assignment_bits[None, :, :] == 1,
        log_p[:, None, atoms],
        0.0,
    ), axis=2)

    safe_count = np.clip(free_count, 0, cardinality)
    pattern_base = atom_scores + log_e[:, safe_count]
    pattern_base[:, ~valid_pattern] = -np.inf

    consistent = (positive & negative) == 0

    compatible = (
        consistent[:, :, None]
        & (
            (positive[:, :, None] & assignments[None, None, :])
            == positive[:, :, None]
        )
        & (
            (negative[:, :, None] & assignments[None, None, :])
            == 0
        )
    )

    theory_log_z = logsumexp(
        np.where(compatible, pattern_base[:, None, :], -np.inf),
        axis=2,
    )
    log_w = logsumexp(
        np.where(compatible, log_q[:, :, None], -np.inf),
        axis=1,
    )
    log_z = logsumexp(pattern_base + log_w, axis=1)

    if not np.isfinite(log_z).all():
        raise DistributionError("Nema dopustive kvantno-logicke mase.")

    pattern_mass = np.exp(
        pattern_base + log_w - log_z[:, None]
    )
    theory_posterior = np.exp(
        log_q + theory_log_z - log_z[:, None]
    )

    require(
        np.allclose(
            pattern_mass.sum(axis=1), 1.0,
            atol=2e-8, rtol=0.0,
        )
        and np.allclose(
            theory_posterior.sum(axis=1), 1.0,
            atol=2e-8, rtol=0.0,
        )
        and np.all(
            log_z + math.lgamma(cardinality + 1) <= 2e-8
        ),
        "Kvantno-logicka raspodela nije normalizovana.",
    )

    marginals = None

    if need_marginals:
        marginals = np.zeros((batch, states))
        marginals[:, atoms] = pattern_mass @ assignment_bits

        count_mass = np.zeros((batch, cardinality + 1))

        for size in range(cardinality + 1):
            count_mass[:, size] = pattern_mass[
                :, free_count == size
            ].sum(axis=1)

        marginals[:, free] = np.einsum(
            "br,brv->bv", count_mass, free_inclusion
        )

        require(
            np.all(marginals >= 0.0)
            and np.all(marginals <= 1.0 + 2e-8)
            and np.allclose(
                marginals.sum(axis=1), float(cardinality),
                atol=2e-8, rtol=0.0,
            ),
            "Ukljucenosti brojeva nisu normalizovane.",
        )

    return {
        "log_q": log_q,
        "log_p": log_p,
        "log_z": log_z,
        "log_w": log_w,
        "theory_log_z": theory_log_z,
        "theory_posterior": theory_posterior,
        "marginals": marginals,
        "compatible": compatible,
        "consistent": consistent,
        "assignments": assignments,
        "assignment_bits": assignment_bits,
        "atom_scores": atom_scores,
        "free_count": free_count,
        "valid_pattern": valid_pattern,
        "atoms": atoms,
        "free": free,
        "cardinality": cardinality,
    }


def target_scores(distribution, targets):
    targets = np.asarray(targets, dtype=np.int64)
    batch, states = distribution["log_p"].shape

    require(
        targets.shape == (batch, distribution["cardinality"])
        and np.all((targets >= 0) & (targets < states))
        and np.all(np.diff(targets, axis=1) > 0),
        "Neispravni ciljni skupovi.",
    )

    atom_bits = np.any(
        targets[:, :, None]
        == distribution["atoms"][None, None, :],
        axis=1,
    )
    assignment = (
        atom_bits
        * (1 << np.arange(len(distribution["atoms"])))[None, :]
    ).sum(axis=1).astype(np.int64)

    log_w_target = distribution["log_w"][
        np.arange(batch), assignment
    ]
    base_score = distribution["log_p"][
        np.arange(batch)[:, None], targets
    ].sum(axis=1)

    log_probability = (
        base_score + log_w_target - distribution["log_z"]
    )

    require(
        not np.isnan(log_probability).any()
        and np.all(log_probability <= 2e-8),
        "Neispravna ciljna verovatnoca.",
    )

    return log_probability, assignment, log_w_target


def exact_map(distribution, row):
    free = distribution["free"]
    atoms = distribution["atoms"]
    log_p = distribution["log_p"][row]

    order = np.lexsort((free, -log_p[free]))
    ordered_free = free[order]
    prefix = np.concatenate([
        np.array([0.0]),
        np.cumsum(log_p[ordered_free]),
    ])

    scores = np.full(len(distribution["assignments"]), -np.inf)

    for pattern in np.flatnonzero(distribution["valid_pattern"]):
        size = int(distribution["free_count"][pattern])
        scores[pattern] = (
            distribution["atom_scores"][row, pattern]
            + prefix[size]
            + distribution["log_w"][row, pattern]
            - distribution["log_z"][row]
        )

    best_score = float(np.max(scores))

    if not np.isfinite(best_score):
        raise DistributionError("Nema dopustivog globalnog maksimuma.")

    candidates = []

    for pattern in np.flatnonzero(scores == best_score):
        selected_atoms = atoms[
            distribution["assignment_bits"][pattern].astype(bool)
        ]
        size = int(distribution["free_count"][pattern])
        combination = np.sort(np.concatenate([
            selected_atoms, ordered_free[:size]
        ]))
        candidates.append((tuple(combination.tolist()), int(pattern)))

    combination, pattern = min(candidates)

    require(
        len(set(combination)) == distribution["cardinality"]
        and best_score <= 2e-8,
        "Neispravan logicki MAP.",
    )

    return {
        "combination": np.asarray(combination, dtype=np.int64),
        "pattern": pattern,
        "log_probability": best_score,
    }


def initial_parameters(feature_count, restart):
    index = np.arange(
        GATE_COUNT * feature_count, dtype=float
    ).reshape(GATE_COUNT, feature_count)

    theta = (
        0.025 * np.sin(index + 1.0 + 0.71 * restart)
        / np.sqrt(feature_count)
    )

    gates = np.arange(1, GATE_COUNT + 1, dtype=float)
    theta[:, 0] = (
        np.pi / 2.0
        + 0.17 * np.sin(gates)
        + 0.20 * restart * np.cos(gates * np.sqrt(2.0))
    )

    return theta.ravel()


def regularization_mask(feature_count):
    mask = np.ones((GATE_COUNT, feature_count))
    mask[:, 0] = 0.0
    return mask.ravel()


def model_forward(
    theta, features, graph, positive, negative,
    need_marginals=True,
):
    weights = theta.reshape(GATE_COUNT, features.shape[1])

    with np.errstate(over="ignore", invalid="ignore"):
        angles = features @ weights.T

    if not np.isfinite(angles).all():
        raise DistributionError("Nekonacni uglovi kvantnih kapija.")

    amplitudes = np.stack([
        np.cos(angles / 2.0),
        np.sin(angles / 2.0),
    ], axis=2)

    theory_logs = register_logs(
        amplitudes[:, :THEORY_QUBITS], THEORY_REGISTER
    )[:, :len(graph["masks"])]

    number_logs = register_logs(
        amplitudes[:, THEORY_QUBITS:], NUMBER_REGISTER
    )[:, :N]

    distribution = logical_distribution(
        theory_logs,
        number_logs,
        positive,
        negative,
        graph["atoms"],
        need_marginals=need_marginals,
    )

    return {
        "amplitudes": amplitudes,
        "distribution": distribution,
    }


def angle_gradient_from_log_masses(
    log_mass_gradient, raw_bits, amplitudes
):
    batch = len(log_mass_gradient)
    qubits = raw_bits.shape[1]
    bit_gradient = np.zeros((batch, qubits, 2))

    for bit in range(qubits):
        for value in (0, 1):
            bit_gradient[:, bit, value] = log_mass_gradient[
                :, raw_bits[:, bit] == value
            ].sum(axis=1)

    derivatives = np.stack([
        -0.5 * amplitudes[:, :, 1],
        0.5 * amplitudes[:, :, 0],
    ], axis=2)
    contributions = np.zeros_like(amplitudes)

    with np.errstate(
        divide="ignore", invalid="ignore", over="ignore"
    ):
        np.divide(
            2.0 * bit_gradient * derivatives,
            amplitudes,
            out=contributions,
            where=amplitudes != 0.0,
        )

    return contributions.sum(axis=2)


def objective(
    theta, features, targets, graph, closure, weight,
    with_gradient=True,
):
    require(
        len(features) == len(targets) > 0,
        "Neispravan trening skup.",
    )

    if not np.isfinite(theta).all():
        return np.inf, np.zeros_like(theta)

    gradient = np.zeros((GATE_COUNT, features.shape[1]))
    loss_sum = 0.0

    for start in range(0, len(targets), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(targets))
        batch_features = features[start:stop]
        batch_targets = targets[start:stop]

        try:
            forward = model_forward(
                theta,
                batch_features,
                graph,
                closure["positive"][start:stop],
                closure["negative"][start:stop],
                need_marginals=with_gradient,
            )
        except DistributionError:
            return np.inf, np.zeros_like(theta)

        distribution = forward["distribution"]
        log_probability, assignment, log_w_target = target_scores(
            distribution, batch_targets
        )

        if not np.isfinite(log_probability).all():
            return np.inf, np.zeros_like(theta)

        loss_sum += float(
            np.maximum(-log_probability, 0.0).sum()
        )

        if not with_gradient:
            continue

        batch = stop - start
        theory_count = len(graph["masks"])

        accepts_target = distribution["compatible"][
            np.arange(batch)[:, None],
            np.arange(theory_count)[None, :],
            assignment[:, None],
        ]

        target_theory_posterior = np.exp(np.where(
            accepts_target,
            distribution["log_q"] - log_w_target[:, None],
            -np.inf,
        ))

        require(
            np.allclose(
                target_theory_posterior.sum(axis=1),
                1.0,
                atol=2e-8,
                rtol=0.0,
            ),
            "Ciljna teorijska raspodela nije normalizovana.",
        )

        theory_gradient = (
            distribution["theory_posterior"]
            - target_theory_posterior
        )

        number_gradient = distribution["marginals"].copy()
        number_gradient[
            np.arange(batch)[:, None], batch_targets
        ] -= 1.0

        theory_raw_bits = THEORY_REGISTER["bits"][
            THEORY_REGISTER["inverse"][:theory_count]
        ]
        number_raw_bits = NUMBER_REGISTER["bits"][
            NUMBER_REGISTER["inverse"][:N]
        ]

        theory_angles = angle_gradient_from_log_masses(
            theory_gradient,
            theory_raw_bits,
            forward["amplitudes"][:, :THEORY_QUBITS],
        )
        number_angles = angle_gradient_from_log_masses(
            number_gradient,
            number_raw_bits,
            forward["amplitudes"][:, THEORY_QUBITS:],
        )
        angle_gradient = np.concatenate([
            theory_angles, number_angles
        ], axis=1)

        if not np.isfinite(angle_gradient).all():
            return np.inf, np.zeros_like(theta)

        gradient += angle_gradient.T @ batch_features

    mask = regularization_mask(features.shape[1])

    with np.errstate(over="ignore", invalid="ignore"):
        loss = (
            loss_sum / len(targets)
            + weight * np.dot(mask * theta, theta) / GATE_COUNT
        )
        full_gradient = (
            gradient.ravel() / len(targets)
            + 2.0 * weight * mask * theta / GATE_COUNT
        )

    if not np.isfinite(loss) or not np.isfinite(full_gradient).all():
        return np.inf, np.zeros_like(theta)

    return float(loss), full_gradient


def fit_model(features, targets, graph, closure, weight):
    best = None

    for restart in range(RESTARTS):
        tracked = {"loss": np.inf, "theta": None, "calls": 0}

        def tracked_objective(candidate):
            value, gradient = objective(
                candidate, features, targets, graph, closure, weight
            )
            tracked["calls"] += 1

            if np.isfinite(value) and value < tracked["loss"]:
                tracked["loss"] = value
                tracked["theta"] = candidate.copy()

            if DIAGNOSTICS and tracked["calls"] % 20 == 0:
                detail(
                    f"Pocetak={restart + 1}, "
                    f"procena={tracked['calls']}, "
                    f"NLL+L2={tracked['loss']:.9f}"
                )

            return value, gradient

        result = minimize(
            tracked_objective,
            initial_parameters(features.shape[1], restart),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": GRADIENT_TOLERANCE,
                "ftol": 1e-10,
                "maxls": 30,
                "maxcor": 10,
            },
        )

        require(
            tracked["theta"] is not None,
            "Optimizator nije nasao konacan model.",
        )

        theta = tracked["theta"]
        loss, gradient = objective(
            theta, features, targets, graph, closure, weight
        )
        norm = float(np.max(np.abs(gradient)))

        candidate = {
            "theta": theta.copy(),
            "loss": loss,
            "gradient": norm,
            "converged": norm <= GRADIENT_TOLERANCE,
            "message": str(result.message),
        }

        if best is None or loss < best["loss"]:
            best = candidate

    return best


def slice_closure(closure, selection):
    return {
        name: values[selection]
        for name, values in closure.items()
    }


def evaluate(theta, features, targets, graph, closure):
    losses = []
    hits = []

    for start in range(0, len(targets), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(targets))

        forward = model_forward(
            theta,
            features[start:stop],
            graph,
            closure["positive"][start:stop],
            closure["negative"][start:stop],
            need_marginals=False,
        )
        distribution = forward["distribution"]
        log_probability, _, _ = target_scores(
            distribution, targets[start:stop]
        )
        losses.extend(
            np.maximum(-log_probability, 0.0).tolist()
        )

        for row, target in enumerate(targets[start:stop]):
            prediction = exact_map(distribution, row)["combination"]
            hits.append(
                len(set(prediction.tolist()) & set(target.tolist()))
            )

    require(len(losses) > 0, "Prazna evaluacija.")
    losses = np.asarray(losses)

    return {
        "nll": float(losses.mean()),
        "hits": np.asarray(hits, dtype=np.int64),
        "zero_targets": int(np.isposinf(losses).sum()),
    }


def qiskit_register(angles, description):
    qubits = len(angles)
    circuit = QuantumCircuit(qubits)

    for bit in range(qubits):
        circuit.ry(float(angles[bit]), bit)

    for control in range(qubits - 1):
        circuit.cx(control, control + 1)

    actual = np.asarray(
        Statevector.from_instruction(circuit).data
    )

    raw_amplitudes = np.prod(np.where(
        description["bits"] == 1,
        np.sin(angles / 2.0)[None, :],
        np.cos(angles / 2.0)[None, :],
    ), axis=1)

    require(
        np.allclose(
            actual,
            raw_amplitudes[description["inverse"]],
            atol=2e-10,
            rtol=2e-9,
        ),
        "Qiskit pripremne amplitude nisu dosledne.",
    )

    return actual


def verify_logic_oracle(angles, accepted, source_state):
    require(
        accepted.shape == (THEORY_DIM,),
        "Neispravan predikat logickog orakla.",
    )

    circuit = QuantumCircuit(THEORY_QUBITS + 1)

    for bit in range(THEORY_QUBITS):
        circuit.ry(float(angles[bit]), bit)

    for control in range(THEORY_QUBITS - 1):
        circuit.cx(control, control + 1)

    flag = THEORY_QUBITS
    diagonal = np.ones(2 * THEORY_DIM, dtype=np.complex128)
    diagonal[THEORY_DIM:] = 1.0 - 2.0 * accepted.astype(float)

    circuit.h(flag)
    circuit.append(
        DiagonalGate(diagonal.tolist()),
        range(THEORY_QUBITS + 1),
    )
    circuit.h(flag)

    actual = np.asarray(
        Statevector.from_instruction(circuit).data
    ).reshape(2, THEORY_DIM)

    expected = np.zeros_like(actual)
    expected[accepted.astype(np.int64), np.arange(THEORY_DIM)] = (
        source_state
    )

    require(
        np.allclose(
            actual, expected,
            atol=2e-9, rtol=2e-9,
        ),
        "Qiskit logicki orakl ne oznacava tacne teorije.",
    )

    return float(np.sum(np.abs(actual[1]) ** 2))


def qiskit_final(theta, feature, graph, pattern, budget):
    closure = compile_closure(
        graph, pattern[None, :], budget
    )

    # Nezavisna provera svih formalnih izvoda na aktuelnom kontekstu.
    for theory_index in range(len(graph["masks"])):
        records = derive_certificate(
            graph, pattern, theory_index, budget
        )
        positive, negative = verify_certificate(
            graph, pattern, theory_index, budget, records
        )

        require(
            positive == closure["positive"][0, theory_index]
            and negative == closure["negative"][0, theory_index],
            "Kompajlirani i sertifikovani izvodi se razlikuju.",
        )

    weights = theta.reshape(GATE_COUNT, len(feature))
    angles = weights @ feature

    theory_state = qiskit_register(
        angles[:THEORY_QUBITS], THEORY_REGISTER
    )
    number_state = qiskit_register(
        angles[THEORY_QUBITS:], NUMBER_REGISTER
    )

    theory_count = len(graph["masks"])
    log_q = log_amplitudes(theory_state[:theory_count])[None, :]
    log_p = log_amplitudes(number_state[:N])[None, :]

    distribution = logical_distribution(
        log_q,
        log_p,
        closure["positive"],
        closure["negative"],
        graph["atoms"],
    )
    prediction = exact_map(distribution, 0)

    consistent_flag = np.zeros(THEORY_DIM, dtype=bool)
    consistent_flag[:theory_count] = distribution["consistent"][0]

    verify_logic_oracle(
        angles[:THEORY_QUBITS],
        consistent_flag,
        theory_state,
    )

    accepts_next = np.zeros(THEORY_DIM, dtype=bool)
    accepts_next[:theory_count] = distribution["compatible"][
        0, :, prediction["pattern"]
    ]

    actual_accepted_mass = verify_logic_oracle(
        angles[:THEORY_QUBITS],
        accepts_next,
        theory_state,
    )

    require(
        np.isclose(
            actual_accepted_mass,
            np.exp(distribution["log_w"][0, prediction["pattern"]]),
            atol=2e-9,
            rtol=2e-8,
        ),
        "Qiskit masa logickih tumacenja nije dosledna.",
    )

    numerical = model_forward(
        theta,
        feature[None, :],
        graph,
        closure["positive"],
        closure["negative"],
    )["distribution"]

    require(
        np.allclose(
            distribution["marginals"],
            numerical["marginals"],
            atol=2e-9,
            rtol=2e-8,
        )
        and np.array_equal(
            prediction["combination"],
            exact_map(numerical, 0)["combination"],
        ),
        "Zavrsni Qiskit i numericki model se razlikuju.",
    )

    prediction["marginals"] = distribution["marginals"][0]
    prediction["log_acceptance"] = (
        math.lgamma(K + 1) + distribution["log_z"][0]
    )

    return prediction


def self_checks():
    atom = Atom(FUTURE, 17, 1)
    rule = Rule(Atom(PAST_ONE, 3, 0), atom)

    require(
        godel_decode(atom.code) == atom.tokens()
        and godel_decode(rule.code) == rule.tokens()
        and atom.code != rule.code,
        "Aritmeticko kodiranje nije jednoznacno.",
    )

    # Rucna teorija proverava pozitivno, negativno,
    # nerazreseno, sukob i zavisnost od dubine.
    atoms = np.array([2, 5, 8, 11, 14, 17])
    conditions = [None] * (2 * len(atoms))
    conditions[0] = (1, 1)
    conditions[1] = (0, 1)

    graph = assemble_graph(
        atoms,
        conditions,
        [(0, 1, 1, 0), (1, 0, 2, 1)],
    )

    pattern = np.zeros(2 * N + 1, dtype=bool)
    pattern[[0, 1, 3, 7, 13, 20, 30]] = True

    shallow = compile_closure(graph, pattern[None, :], 2)
    deep = compile_closure(graph, pattern[None, :], 4)

    empty_index = int(np.flatnonzero(graph["masks"] == 0)[0])
    positive_index = int(np.flatnonzero(
        graph["masks"] == (1 << 1)
    )[0])
    conflict_index = int(np.flatnonzero(
        graph["masks"] == ((1 << 0) | (1 << 1))
    )[0])

    require(
        shallow["positive"][0, empty_index] == 0
        and shallow["negative"][0, empty_index] == 0
        and shallow["positive"][0, positive_index] == 1
        and shallow["negative"][0, positive_index] == 0
        and deep["positive"][0, positive_index] == 5
        and deep["negative"][0, positive_index] == 2
        and (
            deep["positive"][0, conflict_index]
            & deep["negative"][0, conflict_index]
        ) != 0,
        "Tri statusa ili dokazna dubina nisu ispravni.",
    )

    for budget in PROOF_BUDGETS:
        compiled = compile_closure(graph, pattern[None, :], budget)

        for theory_index in range(len(graph["masks"])):
            certificate = derive_certificate(
                graph, pattern, theory_index, budget
            )
            positive, negative = verify_certificate(
                graph, pattern, theory_index, budget, certificate
            )

            require(
                positive == compiled["positive"][0, theory_index]
                and negative == compiled["negative"][0, theory_index],
                "Dokazni kompajler i proveravac se razlikuju.",
            )

    records = list(derive_certificate(
        graph, pattern, positive_index, 4
    ))
    corrupted_index = next(
        index for index, record in enumerate(records)
        if record[2] != 0
    )
    head, depth, code, premise = records[corrupted_index]
    records[corrupted_index] = (
        head, depth, code + 1, premise
    )

    try:
        verify_certificate(
            graph, pattern, positive_index, 4, records
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Neispravan sertifikat nije odbijen.")

    # Potpuna provera male raspodele preko svih kombinacija.
    small_p = np.exp(
        0.3 * np.sin(np.arange(5, dtype=float))
    )
    small_p /= small_p.sum()
    small_q = np.array([0.35, 0.25, 0.25, 0.15])

    positive = np.array([[0, 1, 0, 1]], dtype=np.int64)
    negative = np.array([[0, 0, 2, 1]], dtype=np.int64)
    small_atoms = np.array([1, 3])

    distribution = logical_distribution(
        np.log(small_q)[None, :],
        np.log(small_p)[None, :],
        positive,
        negative,
        small_atoms,
        cardinality=2,
    )

    combinations = list(itertools.combinations(range(5), 2))
    unnormalized = []

    for combination in combinations:
        assignment = sum(
            (1 << index)
            for index, number in enumerate(small_atoms)
            if number in combination
        )
        weight = 0.0

        for theory in range(len(small_q)):
            pos = int(positive[0, theory])
            neg = int(negative[0, theory])

            if (
                (pos & neg) == 0
                and (assignment & pos) == pos
                and (assignment & neg) == 0
            ):
                weight += small_q[theory]

        unnormalized.append(
            np.prod(small_p[list(combination)]) * weight
        )

    unnormalized = np.asarray(unnormalized)
    probabilities = unnormalized / unnormalized.sum()
    brute_marginals = np.zeros(5)

    for combination, mass in zip(combinations, probabilities):
        brute_marginals[list(combination)] += mass

    prediction = exact_map(distribution, 0)
    predicted_index = combinations.index(
        tuple(prediction["combination"].tolist())
    )

    require(
        np.isclose(
            distribution["log_z"][0],
            np.log(unnormalized.sum()),
            atol=1e-12,
        )
        and np.allclose(
            distribution["marginals"][0],
            brute_marginals,
            atol=1e-12,
            rtol=0.0,
        )
        and np.isclose(
            probabilities[predicted_index],
            probabilities.max(),
            atol=1e-12,
        )
        and distribution["theory_posterior"][0, 3] == 0.0,
        "Potpuna provera logicke raspodele nije prosla.",
    )

    fixture = np.array([
        [0, 1, 7, 13, 20, 30, 38],
        [1, 2, 8, 14, 22, 31, 37],
        [0, 5, 8, 16, 24, 33, 38],
        [2, 7, 11, 18, 23, 29, 36],
        [1, 8, 14, 19, 26, 32, 37],
        [3, 9, 15, 21, 27, 34, 38],
        [0, 6, 13, 18, 25, 31, 36],
        [2, 8, 12, 19, 24, 32, 38],
    ], dtype=np.int64)

    times = np.arange(1, len(fixture))
    patterns = history_patterns(fixture, times)
    targets = fixture[1:]
    cutoff = 5

    transform = fit_transform(patterns, cutoff)
    features = transform_features(patterns, transform)
    closure = compile_closure(graph, patterns, 4)
    theta = initial_parameters(features.shape[1], 0)

    value, analytic = objective(
        theta,
        features[:cutoff],
        targets[:cutoff],
        graph,
        slice_closure(closure, slice(0, cutoff)),
        REGULARIZATIONS[0],
    )
    require(np.isfinite(value), "Nekonacan proverni gubitak.")

    selected = set(
        np.linspace(0, len(theta) - 1, 18, dtype=int).tolist()
    )

    for gate in (0, THEORY_QUBITS - 1, THEORY_QUBITS, GATE_COUNT - 1):
        selected.add(gate * features.shape[1])

    step = 1e-6

    for index in sorted(selected):
        upper = theta.copy()
        lower = theta.copy()
        upper[index] += step
        lower[index] -= step

        numeric = (
            objective(
                upper, features[:cutoff], targets[:cutoff], graph,
                slice_closure(closure, slice(0, cutoff)),
                REGULARIZATIONS[0], with_gradient=False,
            )[0]
            - objective(
                lower, features[:cutoff], targets[:cutoff], graph,
                slice_closure(closure, slice(0, cutoff)),
                REGULARIZATIONS[0], with_gradient=False,
            )[0]
        ) / (2.0 * step)

        require(
            np.isclose(
                analytic[index], numeric,
                atol=3e-5, rtol=4e-4,
            ),
            "Koordinatna provera gradijenta nije prosla.",
        )

    for direction_index in range(3):
        direction = np.sin(
            (np.arange(len(theta)) + 1.0)
            * (0.31 + 0.17 * direction_index)
        )
        direction /= np.linalg.norm(direction)

        numeric = (
            objective(
                theta + step * direction,
                features[:cutoff], targets[:cutoff], graph,
                slice_closure(closure, slice(0, cutoff)),
                REGULARIZATIONS[0], with_gradient=False,
            )[0]
            - objective(
                theta - step * direction,
                features[:cutoff], targets[:cutoff], graph,
                slice_closure(closure, slice(0, cutoff)),
                REGULARIZATIONS[0], with_gradient=False,
            )[0]
        ) / (2.0 * step)

        require(
            np.isclose(
                analytic @ direction, numeric,
                atol=3e-5, rtol=4e-4,
            ),
            "Usmerena provera gradijenta nije prosla.",
        )

    qiskit_final(
        theta, features[cutoff], graph, patterns[cutoff], 4
    )

    learned = learn_graph(patterns[:cutoff], targets[:cutoff])
    changed = fixture.copy()
    changed[cutoff + 1:] = changed[cutoff + 1:][::-1]
    changed_patterns = history_patterns(changed, times)
    changed_graph = learn_graph(
        changed_patterns[:cutoff],
        changed[1:cutoff + 1],
    )
    changed_transform = fit_transform(
        changed_patterns, cutoff
    )

    def graph_signature(value):
        return (
            tuple(value["atoms"].tolist()),
            tuple(
                rule.code if rule is not None else 0
                for rule in value["roots"]
            ),
            tuple(rule.code for rule in value["link_rules"]),
        )

    require(
        np.array_equal(
            patterns[:cutoff + 1],
            changed_patterns[:cutoff + 1],
        )
        and graph_signature(learned) == graph_signature(changed_graph)
        and np.array_equal(
            transform["keep"], changed_transform["keep"]
        )
        and np.allclose(
            transform["mean"], changed_transform["mean"]
        )
        and np.allclose(
            transform["scale"], changed_transform["scale"]
        ),
        "Buduci podaci uticu na raniju teoriju ili skaliranje.",
    )

    detail("Ugradjene kodiranje, dokaz, Qiskit, gradijent i MAP provere su prosle.")


def longest_consecutive(values):
    longest = 1
    current = 1

    for difference in np.diff(values):
        current = current + 1 if difference == 1 else 1
        longest = max(longest, current)

    return longest


def process_dataset(dataset):
    data = dataset["data"]
    name = dataset["path"].name
    targets = data[1:]
    times = np.arange(1, len(data))
    patterns = history_patterns(data, times)
    total = len(targets)

    train_end = int(TRAIN_FRACTION * total)
    validation_end = int(VALIDATION_END_FRACTION * total)

    require(
        0 < train_end < validation_end < total,
        f"{name}: nedovoljno podataka za hronolosku podelu.",
    )

    print(f"{name} | ucenje teorija i amplituda", flush=True)
    detail(f"Redova={len(data)}, SHA256={dataset['digest']}")

    graph = learn_graph(patterns[:train_end], targets[:train_end])
    transform = fit_transform(patterns, train_end)
    features = transform_features(patterns, transform)
    best = None

    for budget in PROOF_BUDGETS:
        closure = compile_closure(graph, patterns, budget)

        for weight in REGULARIZATIONS:
            fit = fit_model(
                features[:train_end],
                targets[:train_end],
                graph,
                slice_closure(closure, slice(0, train_end)),
                weight,
            )

            try:
                metrics = evaluate(
                    fit["theta"],
                    features[train_end:validation_end],
                    targets[train_end:validation_end],
                    graph,
                    slice_closure(
                        closure, slice(train_end, validation_end)
                    ),
                )
            except DistributionError as exc:
                detail(f"Kandidat odbijen: {exc}")
                continue

            detail(
                f"Budzet={budget}, L2={weight}, "
                f"validacioni NLL={metrics['nll']:.9f}"
            )

            if np.isfinite(metrics["nll"]) and (
                best is None or metrics["nll"] < best["metrics"]["nll"]
            ):
                best = {
                    "budget": budget,
                    "weight": weight,
                    "fit": fit,
                    "metrics": metrics,
                }

    require(best is not None, f"{name}: nema validnog modela.")

    print(
        f"Validacija | dubina={best['budget']}, L2={best['weight']}, "
        f"NLL={best['metrics']['nll']:.6f}, "
        f"pogodaka={best['metrics']['hits'].mean():.4f}",
        flush=True,
    )

    test_graph = learn_graph(
        patterns[:validation_end], targets[:validation_end]
    )
    test_transform = fit_transform(patterns, validation_end)
    test_features = transform_features(patterns, test_transform)
    test_closure = compile_closure(
        test_graph, patterns, best["budget"]
    )

    test_fit = fit_model(
        test_features[:validation_end],
        targets[:validation_end],
        test_graph,
        slice_closure(test_closure, slice(0, validation_end)),
        best["weight"],
    )
    test_metrics = evaluate(
        test_fit["theta"],
        test_features[validation_end:],
        targets[validation_end:],
        test_graph,
        slice_closure(test_closure, slice(validation_end, None)),
    )

    print(
        f"Test | NLL={test_metrics['nll']:.6f}, "
        f"pogodaka={test_metrics['hits'].mean():.4f}",
        flush=True,
    )

    if test_metrics["zero_targets"]:
        print(
            f"Upozorenje: test ciljevi sa nultom masom="
            f"{test_metrics['zero_targets']}"
        )

    # Hiperparametri su zakljucani pre testa.
    # Zavrsna teorija i amplitude koriste sve istorijske prelaze.
    final_graph = learn_graph(patterns, targets)
    final_transform = fit_transform(patterns, total)
    final_features = transform_features(patterns, final_transform)
    final_closure = compile_closure(
        final_graph, patterns, best["budget"]
    )

    final_fit = fit_model(
        final_features,
        targets,
        final_graph,
        final_closure,
        best["weight"],
    )

    next_patterns = history_patterns(
        data, np.array([len(data)])
    )
    next_features = transform_features(
        next_patterns, final_transform
    )

    result = qiskit_final(
        final_fit["theta"],
        next_features[0],
        final_graph,
        next_patterns[0],
        best["budget"],
    )
    combination = result["combination"] + 1

    require(
        len(combination) == K
        and len(set(combination.tolist())) == K
        and np.all((combination >= 1) & (combination <= N)),
        "Nelegalna NEXT kombinacija.",
    )

    unconfirmed = [
        label
        for label, fit in (
            ("validacioni", best["fit"]),
            ("test", test_fit),
            ("zavrsni", final_fit),
        )
        if not fit["converged"]
    ]

    if unconfirmed:
        print(
            "Napomena: mali gradijent nije potvrdjen za "
            + ", ".join(unconfirmed)
            + " model."
        )

    if all(rule is None for rule in final_graph["roots"]):
        print(
            "Napomena: nema podrzanih istorijskih premisa; "
            "buduci literali ostaju nerazreseni."
        )

    if np.array_equal(combination, data[-1] + 1):
        print("Upozorenje: NEXT ponavlja poslednje kolo.")

    run = longest_consecutive(combination)
    if run >= 5:
        print(f"Upozorenje: NEXT ima {run} uzastopnih brojeva.")

    span = int(combination[-1] - combination[0])
    span_percentile = float(
        np.mean((data[:, -1] - data[:, 0]) <= span)
    )

    if span_percentile <= 0.01:
        print("Upozorenje: izrazito grupisana NEXT kombinacija.")

    detail(
        f"Log prihvacene mase={result['log_acceptance']:.9f}, "
        f"log P(NEXT)={result['log_probability']:.9f}"
    )

    print(
        f"{name} NEXT: "
        + " ".join(str(int(value)) for value in combination),
        flush=True,
    )

    return {
        "name": name,
        "digest": dataset["digest"],
        "next": combination,
        "marginals": result["marginals"],
        "log_probability": result["log_probability"],
    }


def main():
    print(
        "Quantum Godel Proof Extension V1 | "
        "sertifikovani izvodi, nerazreseni literali i saglasna prosirenja"
    )
    print(
        f"Qiskit {qiskit.__version__} | "
        "7 teorijskih + 1 orakl kubit; "
        "7 izlaznih registara po 6 kubita, faktorisana simulacija"
    )

    self_checks()

    results = [
        process_dataset(load_csv(path))
        for path in FILES
    ]

    for first, second in itertools.combinations(results, 2):
        if first["digest"] == second["digest"]:
            print(
                f"Upozorenje: isti sadrzaj "
                f"{first['name']} i {second['name']}."
            )

        if np.array_equal(first["next"], second["next"]):
            marginal_difference = float(np.max(np.abs(
                first["marginals"] - second["marginals"]
            )))
            probability_difference = abs(
                first["log_probability"] - second["log_probability"]
            )

            print(
                f"Isti NEXT: {first['name']} / {second['name']} | "
                f"razlika ukljucenosti={marginal_difference:.3e}, "
                f"log mase NEXT={probability_difference:.3e}"
            )


if __name__ == "__main__":
    main()



"""
Quantum Godel Proof Extension V1 | sertifikovani izvodi, nerazreseni literali i saglasna prosirenja
Qiskit 1.4.4 | 7 teorijskih + 1 orakl kubit; 7 izlaznih registara po 6 kubita, faktorisana simulacija

loto7_4716_k89.csv | ucenje teorija i amplituda
Validacija | dubina=4, L2=0.1, NLL=16.631949, pogodaka=1.3239
Test | NLL=16.665829, pogodaka=1.2444
Napomena: mali gradijent nije potvrdjen za validacioni, test, zavrsni model.
loto7_4716_k89.csv NEXT: 7 x 22 y 24 z 39

loto7_4716_k89_loto_2980.csv | ucenje teorija i amplituda
Validacija | dubina=4, L2=0.01, NLL=16.690904, pogodaka=1.2640
Test | NLL=16.696168, pogodaka=1.2550
Napomena: mali gradijent nije potvrdjen za validacioni, test, zavrsni model.
loto7_4716_k89_loto_2980.csv NEXT: 11 x 15 y 21 z 37

loto7_4716_k89_loto_plus_1736.csv | ucenje teorija i amplituda
Validacija | dubina=2, L2=0.01, NLL=16.739208, pogodaka=1.2500
Test | NLL=16.675808, pogodaka=1.3602
Napomena: mali gradijent nije potvrdjen za validacioni, test, zavrsni model.
loto7_4716_k89_loto_plus_1736.csv NEXT: 7 x 17 y 25 z 39
"""



"""
Formalna pravila, prati njihove posledice i proširuje sistem na mestima koja ostaju neodređena.

Netačna je tvrdnju da Gödelovim teoremama ništa nije dokazano. 
Izvorni rad sadrži eksplicitne dokaze. 
Pod uslovima teoreme postoji aritmetička rečenica koju posmatrana teorija ne može ni dokazati ni opovrgnuti 
— upravo time je teorija nepotpuna. Gödelov rad, teoreme VI i XI. 

Za Loto sledeća adaptaciju:
1. Formalizovane hipoteze o prelazima. Istorijski uslovi, zaključci i koraci izvođenja imaju jednoznačne kodove. Sistem može da proveri odakle je neki zaključak izveden.
2. Tri odvojena statusa zaključka. U okviru izabrane teorije tvrdnja može biti izvedena, opovrgnuta ili ostati neodređena. Neodređenost zadržava svoju masu u prediktivnoj raspodeli.
3. Učenje proširenja teorije. Istorija određuje koja nova pravila korisno razrešavaju prethodno neodređene slučajeve. Proverava se njihova logička saglasnost sa postojećim pravilima.
4. Qiskit raspodela nad saglasnim teorijama i njihovim izlazima. Uče se amplitude različitih proširenja, a njihova izvođenja određuju masu mogućih kombinacija.

Posebnost modela bila bi struktura dokaza i neodređenih zaključaka. 
Kod završenog Cantor/Turing modela centralni objekti su programi i njihovi izlazi; ovde su to teorije, pravila izvođenja i saglasna tumačenja.

Ograničenu pretragu koja nije našla dokaz nazivam „nerazrešeno u zadatom budžetu”. 
Gödelovska nedokazivost zahteva poseban matematički dokaz.
Model koristi formalna pravila i proverljive izvode, odvojeno beležim nerazrešene slučajeve i učim Qiskit amplitude proširenja. 

Posebnost V1 je raspodela nad dopustivim tumačenjima delimičnih teorija. 
Dokazni sloj može da izvede uključivanje broja, izvede njegovo isključivanje ili ga ostavi nerazrešenim. 
Protivrečna proširenja imaće nultu masu. 
Preostalu masu normalizujem preko svih legalnih kombinacija, a tačan maksimum tražim po logičkim obrascima.
Formalno kodirane tvrdnje, proverljivi izvodi i Qiskit raspodela nad proširenjima teorije. 
Nerazrešeni zaključci zadržavaju masu, a protivrečna proširenja se odbacuju.
"""
