from fastapi.responses import JSONResponse
from knowlang.cli.commands.parse import parse_command
from knowlang.cli.types import ParseCommandArgs
from knowlang.utils import FancyLogger
from fastapi import APIRouter, BackgroundTasks

LOG = FancyLogger(__name__)
router = APIRouter()

@router.post("/parse")
async def parse_command_endpoint(args: ParseCommandArgs, background_tasks: BackgroundTasks):
    LOG.info(f"Received parse command with args: {args}")
    
    background_tasks.add_task(parse_command, args=args)
    
    return JSONResponse(
        content={"status": "success", "message": "Parse command triggered successfully."},
        status_code=200
    )