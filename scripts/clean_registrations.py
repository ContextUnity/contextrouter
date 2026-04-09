import argparse
import asyncio
import sys

import redis.asyncio as aioredis
from contextcore import get_context_unit_logger

from contextrouter.core import get_core_config

logger = get_context_unit_logger("clean_registrations")

_REDIS_PREFIX = "router:registrations"


async def main():
    parser = argparse.ArgumentParser(description="Clean registered graphs from Redis.")
    parser.add_argument(
        "project_id",
        nargs="?",
        default="all",
        help="The project ID to delete (e.g. 'nszu'). Defaults to 'all', which will clear all registrations.",
    )

    args = parser.parse_args()
    config = get_core_config()

    try:
        r = aioredis.from_url(config.redis.url, decode_responses=True)
    except Exception as e:
        logger.error("Failed to connect to Redis: %s", e)
        sys.exit(1)

    try:
        if args.project_id == "all":
            logger.info("Finding all registrations...")
            keys = []
            async for key in r.scan_iter(f"{_REDIS_PREFIX}:*"):
                keys.append(key)
            if not keys:
                logger.info("No registrations found.")
            else:
                await r.delete(*keys)
                logger.info("Deleted %s registration keys.", len(keys))
        else:
            base_key = f"{_REDIS_PREFIX}:{args.project_id}"
            hash_key = f"{_REDIS_PREFIX}:{args.project_id}:hash"
            count = await r.delete(base_key, hash_key)
            if count > 0:
                logger.info("Deleted %s keys for project '%s'.", count, args.project_id)
            else:
                logger.warning("No keys found for project '%s'.", args.project_id)
    finally:
        await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
