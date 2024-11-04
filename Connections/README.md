# README

## Liam Keane and Lazuli Kleinhans

We first introduced a red herring group by sampling a similar word to our first group while ensuring that the red herring was not too similar to the original group.

Second, we introduced a non-semantic group by randomly sampling from a list of languages (previously we had several non-semantic lists like names, genres, and places but our model didn't recognize many of the words so we got rid of those options)

Third, we introduced easier/harder difficulty levels within the same puzzle by having the one of the remaining two groups choose the three most semantically similar words to the seed while the final group's three words had to fall outside of a given similarilty range (in general this leads to harder categories).

Just running connections.py normally should create all puzzle types:

```bash
python3 connections.py
```
