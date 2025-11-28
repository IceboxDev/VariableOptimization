# -*- coding: utf-8 -*-

import artificial_intelligence
import modules

import statistics


if __name__ == "__main__":
    database = modules.Database('.config/personal-433622-62e046c7be64.json', force_refresh=True)
    ai = artificial_intelligence.ArtificialIntelligence(database)
    # ai.train(artificial_intelligence.NeuralNetwork, best_of=100)

    ai.load(artificial_intelligence.NeuralNetwork, '[ 799] Model.pt')
    # for game in database.games:
    #     print(game.score, round(ai.infer(game)), round(ai.infer(game) - game.score))

    # players = ["Iris", "Jaqueline", "Luis", "Mantas", "Maximilian (B)", "Melanie (E)", "Nicolo", "Paul", "Simon (G)", "Yawen"]
    # players = [database.players[name] for name in players]
    # print(ai.infer(Game(None, None, None, *players)))

    import itertools
    import collections
    players = {k: v for k, v in database.players.items() if len([g for g in v.games if g.date.year == 2025]) >= 3}

    scores = collections.defaultdict(list)
    for team in itertools.combinations(players.values(), 5):
        game = modules.Game(None, None, None, *team)
        score = ai.infer(game)
        for player in team:
            scores[player.name].append(score)

    for name, score_list in sorted(scores.items(), key=lambda x: statistics.mean(x[1]), reverse=True):
        print(f'{name}: {round(statistics.mean(score_list), 2)} ± {round(statistics.stdev(score_list), 2)}')



