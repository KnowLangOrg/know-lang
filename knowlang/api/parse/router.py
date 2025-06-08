from fastapi.responses import JSONResponse
from knowlang.cli.commands.parse import parse_command
from knowlang.cli.types import ParseCommandArgs
from knowlang.utils import FancyLogger
from fastapi import APIRouter, BackgroundTasks

LOG = FancyLogger(__name__)
router = APIRouter()


# https://stackoverflow.com/questions/71516140/fastapi-runs-api-calls-in-serial-instead-of-parallel-fashion
# https://stackoverflow.com/questions/67599119/fastapi-asynchronous-background-tasks-blocks-other-requests
# very hacky workaround to allow parse command to run in the background
async def parse_command_hack(args: ParseCommandArgs):
    from asyncio import sleep
    await sleep(1)  # Simulate a small delay to yield control to the event loop for response

    await parse_command(args)

@router.post("/parse")
async def parse_command_endpoint(args: ParseCommandArgs, background_tasks: BackgroundTasks):
    LOG.info(f"Received parse command with args: {args}")
    
    background_tasks.add_task(parse_command_hack, args=args)
    
    return JSONResponse(
        content={"status": "success", "message": "Parse command triggered successfully."},
        status_code=200
    )