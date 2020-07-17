""" Create by Ken at 2020 Jul 17 """
from flask import Flask, request, jsonify
from flask_cors import cross_origin

import dao

app = Flask(__name__)


@app.route('/search', methods=['GET'])
@cross_origin()
def find_document():
    try:
        model_type = request.args.get('modelType')
        doc_name = request.args.get('documentName')
        err, rep = dao.find_representation(model_type, doc_name)

        if err:
            return jsonify({
                'status': 1,
                'msg': str(err)
            })

        return jsonify({
            'status': 0,
            'data': {
                'model_type': model_type,
                'representation': rep
            }
        })
    except Exception as e:
        app.logger.error(e, exc_info=True)
        return 'Internal server error', 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999)
