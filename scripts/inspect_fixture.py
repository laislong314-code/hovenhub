import asyncio
import json
import sys

from app.services.sokkerpro_client import SokkerProClient


async def main():
    if len(sys.argv) != 2:
        print("Uso: python inspect_fixture.py <fixture_id>")
        return

    fixture_id = int(sys.argv[1])

    client = SokkerProClient()
    data = await client.get_fixture(fixture_id)

    print("\nTop-level keys:")
    print(list(data.keys()))

    print("\nChaves disponíveis:")
    print(list(data.keys()))

    bet365_keys = [k for k in data.keys() if "BET365" in k]

    print("\nChaves BET365 encontradas:")
    for k in bet365_keys:
        print("-", k)

    filename = f"fixture_{fixture_id}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nJSON salvo em {filename}")


if __name__ == "__main__":
    asyncio.run(main())
