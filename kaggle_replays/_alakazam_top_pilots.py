"""フーディン(alakazam)デッキの上位パイロットごとの対メタ勝率を算出。"""
import json, os, glob, collections, re

BASE = os.path.dirname(__file__)
_RW = re.compile(r'"rewards"\s*:\s*\[\s*(-?\d+|null)\s*,\s*(-?\d+|null)\s*\]')

# 1) episode -> winner index (rewards) — 先頭チャンクのみ読む(高速化)
def load_rewards():
    ep_win = {}
    for f in glob.glob(os.path.join(BASE, 'replays', 'episode-*-replay.json')):
        eid = os.path.basename(f).split('-')[1]
        try:
            with open(f, encoding='utf-8') as fh:
                head = fh.read(4096)
        except Exception:
            continue
        m = _RW.search(head)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        if a == 'null' or b == 'null':
            continue
        a, b = int(a), int(b)
        if a == b:
            continue
        ep_win[eid] = 0 if a > b else 1
    return ep_win

# 2) (episode, pidx) -> archetype
def load_labels():
    lab = {}
    for line in open(os.path.join(BASE, 'deck_predictor/output/deck_labels.jsonl'), encoding='utf-8'):
        d = json.loads(line)
        lab[(d['episode_id'], d['player_index'])] = d['archetype']
    return lab

# 3) (episode, pidx) -> deck_db record (team_name, rank, deck names)
def load_deckdb():
    db = {}
    for line in open(os.path.join(BASE, 'deck_predictor/output/deck_db.jsonl'), encoding='utf-8'):
        d = json.loads(line)
        db[(d['episode_id'], d['player_index'])] = d
    return db

ep_win = load_rewards()
lab = load_labels()
db = load_deckdb()
print(f"replays with decided winner: {len(ep_win)}")

# per-episode: both sides archetype + team known
# collect alakazam games: pilot -> list of (opp_arch, win_bool, rank, deck_key)
pilot_games = collections.defaultdict(list)   # team_name -> list
pilot_ranks = collections.defaultdict(list)   # team_name -> ranks
pilot_decks = collections.defaultdict(collections.Counter)  # team_name -> deck signature counter

for eid, win_idx in ep_win.items():
    for pidx in (0, 1):
        arch = lab.get((eid, pidx))
        if arch != 'alakazam':
            continue
        opp = lab.get((eid, 1 - pidx))
        if opp is None:
            continue
        rec = db.get((eid, pidx))
        if rec is None:
            continue
        team = rec['team_name']
        won = (win_idx == pidx)
        pilot_games[team].append((opp, won))
        rk = rec.get('rank_at_fetch')
        if rk is not None:
            pilot_ranks[team].append(rk)
        # deck signature (frozenset of card ids sorted tuple)
        sig = tuple(sorted(rec['deck_card_ids']))
        pilot_decks[team][sig] += 1

# rank pilots: require min games, sort by best (min) known rank
MIN_GAMES = 15
cands = []
for team, games in pilot_games.items():
    n = len(games)
    if n < MIN_GAMES:
        continue
    ranks = pilot_ranks.get(team, [])
    best_rank = min(ranks) if ranks else 10**9
    wr = sum(1 for _, w in games if w) / n
    cands.append((team, n, best_rank, wr))

# strongest = best leaderboard rank
cands.sort(key=lambda x: (x[2], -x[3]))
top10 = cands[:10]

# opponent archetype columns (main meta)
META = ['alakazam', 'mega_lucario_ex', 'archaludon_ex', 'crustle', 'dragapult_ex',
        'marnie_grimmsnarl_ex', 'mega_starmie_ex', 'rocket_mewtwo_ex',
        'shirona_garchomp_ex', 'other']

def card_summary(team):
    sig, _ = pilot_decks[team].most_common(1)[0]
    # find a deck_db rec with this sig to get names
    for (eid, pidx), rec in db.items():
        if rec['team_name'] == team and tuple(sorted(rec['deck_card_ids'])) == sig:
            names = rec['deck_card_names']
            # top non-energy-ish cards
            return names, len(pilot_decks[team])
    return {}, len(pilot_decks[team])

out = {'pilots': []}
print("\n" + "=" * 100)
for rank_pos, (team, n, best_rank, wr) in enumerate(top10, 1):
    games = pilot_games[team]
    by_opp = collections.defaultdict(lambda: [0, 0])  # opp -> [wins, total]
    for opp, won in games:
        by_opp[opp][1] += 1
        if won:
            by_opp[opp][0] += 1
    names, n_variants = card_summary(team)
    br = best_rank if best_rank < 10**9 else None
    print(f"\n#{rank_pos}  {team}   games={n}  best_rank={br}  overall_WR={wr*100:.1f}%  deck_variants={n_variants}")
    row = {'pos': rank_pos, 'team': team, 'games': n, 'best_rank': br,
           'overall_wr': round(wr*100,1), 'deck_variants': n_variants,
           'deck': names, 'matchups': {}}
    parts = []
    for m in META:
        w, t = by_opp.get(m, [0,0])
        if t == 0:
            parts.append(f"{m}: -")
        else:
            parts.append(f"{m}: {w}/{t} {w/t*100:.0f}%")
        row['matchups'][m] = {'w': w, 't': t}
    # any opp not in META
    for opp, (w,t) in sorted(by_opp.items(), key=lambda x:-x[1][1]):
        if opp not in META:
            row['matchups'][opp] = {'w': w, 't': t}
    print("   " + " | ".join(parts))
    out['pilots'].append(row)

json.dump(out, open(os.path.join(BASE, '_alakazam_top_pilots_results.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print("\nsaved _alakazam_top_pilots_results.json")
