"""
The ``predictmp`` subcommand allows you to make bulk JSON-to-JSON
predictions using a trained model and its :class:`~allennlp.service.predictors.predictor.Predictor` wrapper.

.. code-block:: bash

    $ allennlp predict --help
    usage: allennlp [command] predictmp [-h]
                                      [--output-file OUTPUT_FILE]
                                      [--batch-size BATCH_SIZE]
                                      [--silent]
                                      [--cuda-device CUDA_DEVICE]
                                      [-o OVERRIDES]
                                      [--include-package INCLUDE_PACKAGE]
                                      [--predictor PREDICTOR]
                                      mp_archive_file archive_file input_file

    Run the specified model against a JSON-lines input file.

    positional arguments:
    archive_file          the archived model to make predictions with
    input_file            path to input file

    optional arguments:
    -h, --help            show this help message and exit
    --output-file OUTPUT_FILE
                            path to output file
    --batch-size BATCH_SIZE
                            The batch size to use for processing
    --silent              do not print output to stdout
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
    -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
    --include-package INCLUDE_PACKAGE
                            additional packages to include
    --predictor PREDICTOR
                            optionally specify a specific predictor to use

    --try-top N
                            run top N results of multi-para through BiDAF (defaults to 1)
"""
import json
import logging
import argparse
from contextlib import ExitStack
import sys
from typing import Optional, IO

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class PredictMP(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the specified model against a JSON-lines input file.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('mp_archive_file', type=str, help='the archived model to find paragraph')
        subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
        subparser.add_argument('input_file', type=argparse.FileType('r'), help='path to input file')

        subparser.add_argument('--output-file', type=argparse.FileType('w'), help='path to output file')
        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')

        subparser.add_argument('--silent', action='store_true', help='do not print output to stdout')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--predictor',
                               type=str,
                               help='optionally specify a specific predictor to use')

        subparser.add_argument('--try-top', type=int, default=1, help='number of top choices from multi-para to run through BiDAF')

        subparser.set_defaults(func=_predict)

        return subparser

def _get_predictors(args: argparse.Namespace) -> list:
    check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_file,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)
    mp_archive = load_archive(args.mp_archive_file,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)
    Predictors = []
    Predictors.append(Predictor.from_archive(mp_archive, args.predictor))
    Predictors.append(Predictor.from_archive(archive))

    return Predictors

def uniquePar():
    s = set()
    def test( item ):
        if item[1] in s: return False
        s.add( item[1] )
        return True
    return test

def top_spans( starts, ends, n ):
# O( N log N ) where N = Np * Lp * (Lp-1) / 2
    assert len(starts) == len(ends)
    return list(filter(
               uniquePar(),
               sorted((
                  (starts[p][i] + ends[p][j], p, i, j)
                    for p in range(len(starts))
                      for i in range(len(starts[p])-1)
                        for j in range(i+1, len(ends[p]))
               ), reverse=True)
            ))[:n]

def format_bidaf( output, par, mp_logit ):
    span = output['best_span']
    span.insert(0, par)

    return {
       'span': span,
       'text': output['best_span_str'],
       'logit': output['span_start_logits'][span[1]] + output['span_end_logits'][span[2]],
       'mp_logit': mp_logit
    }

def _run(predictors: list,
         input_file: IO,
         output_file: Optional[IO],
         batch_size: int,
         print_to_console: bool,
         top_n: 1) -> None:

    predictor = predictors[0]
    bidaf_predictor = predictors[1]

    def _run_predictor(batch_data):
        if len(batch_data) == 1:
            logger.info("Running multi-para BiDAF")
            result = predictor.predict_json(batch_data[0])
            logger.info("Multi-para BiDAF complete")
            # Batch results return a list of json objects, so in
            # order to iterate over the result below we wrap this in a list.
            results = [result]
        else:
            results = predictor.predict_batch_json(batch_data)

        for model_input, output in zip(batch_data, results):
            logger.info( "Model-reported best span %s" % output['best_span'] )

            top = top_spans( output['paragraph_span_start_logits'], output['paragraph_span_end_logits'], top_n )
            logger.info( "Derived top %d spans %s" % (top_n, top) )

            data = [ {
                'question': model_input['question'],
                'passage': model_input['passages'][t[1]]
            } for t in top ]

            data.append( {
                'question': model_input['question'],
                'passage': ' '.join( model_input['passages'] )
            } )

            logger.info("Running BiDAF for %d options" % len(data))
            results = bidaf_predictor.predict_batch_json(data)
            logger.info("BiDAF complete")

            results = {
               'passages': model_input['passages'],
               'question': model_input['question'],
               'MP': { 'span': output['best_span'], 'text': output['best_span_str'], 'logit': top[0][0] },
               'BiDAF': format_bidaf( results[-1], -1, -1 ),
               'MP+BiDAF': [ format_bidaf( results[p], top[p][1], top[p][0] ) for p in range(len(results)-1) ]
            }

            if output_file:
                output_file.write(json.dumps(results))

            print( "Question\t%s" % model_input['question'] )
            print( "MP\t\t%s" % json.dumps( results['MP'] ) )
            print( "BiDAF\t\t%s" % json.dumps( results['BiDAF'] ) )
            print( "MP+BiDAF\t%s" % json.dumps( results['MP+BiDAF'] ) )

    batch_json_data = []
    for line in input_file:
        if not line.isspace():
            # Collect batch size amount of data.
            json_data = predictor.load_line(line)
            batch_json_data.append(json_data)
            if len(batch_json_data) == batch_size:
                _run_predictor(batch_json_data)
                batch_json_data = []

    # We might not have a dataset perfectly divisible by the batch size,
    # so tidy up the scraps.
    if batch_json_data:
        _run_predictor(batch_json_data)


def _predict(args: argparse.Namespace) -> None:
    predictors = _get_predictors(args)
    output_file = None

    if args.silent and not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    # ExitStack allows us to conditionally context-manage `output_file`, which may or may not exist
    with ExitStack() as stack:
        input_file = stack.enter_context(args.input_file)  # type: ignore
        if args.output_file:
            output_file = stack.enter_context(args.output_file)  # type: ignore

        _run(predictors,
             input_file,
             output_file,
             args.batch_size,
             not args.silent,
             args.try_top)
