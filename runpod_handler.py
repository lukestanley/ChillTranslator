import runpod


# Loads model:
from chill import improvement_loop


def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    text = job_input.get('text', 'You guys are so slow, we will never ship it!')
    print("got this input text:",text)
    return str(improvement_loop(text))
    
runpod.serverless.start({"handler": handler})
