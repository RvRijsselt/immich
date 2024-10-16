import { Inject, Injectable } from '@nestjs/common';
import { readFile } from 'node:fs/promises';
import { CLIPConfig } from 'src/dtos/model-config.dto';
import { ILoggerRepository } from 'src/interfaces/logger.interface';
import {
  ClipTextualResponse,
  ClipVisualResponse,
  FaceDetectionOptions,
  FacialRecognitionResponse,
  IMachineLearningRepository,
  MachineLearningRequest,
  ModelPayload,
  ModelTask,
  ModelType,
} from 'src/interfaces/machine-learning.interface';
import { Instrumentation } from 'src/utils/instrumentation';

const errorPrefix = 'Machine learning request';

@Instrumentation()
@Injectable()
export class MachineLearningRepository implements IMachineLearningRepository {
  constructor(@Inject(ILoggerRepository) private logger: ILoggerRepository) {
    this.logger.setContext(MachineLearningRepository.name);
  }

  private async find_working_server(url: string): Promise<string> {
    const paths = url.split(';').map((item) => item.trim());
    for (const path of paths) {
      const r = await fetch(path, { method: 'GET' }).catch((error: Error | any) => {
        this.logger.warn(`${errorPrefix} ping to "${path}" failed with ${error?.cause || error}`);
        return { status: 0, text: 'nop' };
      });
      if (r.status == 200) {
        this.logger.debug(`Response from ${path}`);
        return path;
      }
    }
    throw new Error(`${errorPrefix}: no machine learning server found.`);
  }

  private async predict<T>(url: string, payload: ModelPayload, config: MachineLearningRequest): Promise<T> {
    const formData = await this.getFormData(payload, config);

    this.logger.debug(`Predicting with ${url.split(';')}`);
    const workingurl = await this.find_working_server(url);

    const res = await fetch(new URL('/predict', workingurl), { method: 'POST', body: formData }).catch(
      (error: Error | any) => {
        throw new Error(`${errorPrefix} to "${workingurl}" failed with ${error?.cause || error}`);
      },
    );

    if (res.status >= 400) {
      throw new Error(`${errorPrefix} '${JSON.stringify(config)}' failed with status ${res.status}: ${res.statusText}`);
    }
    return res.json();
  }

  async detectFaces(url: string, imagePath: string, { modelName, minScore }: FaceDetectionOptions) {
    const request = {
      [ModelTask.FACIAL_RECOGNITION]: {
        [ModelType.DETECTION]: { modelName, options: { minScore } },
        [ModelType.RECOGNITION]: { modelName },
      },
    };
    const response = await this.predict<FacialRecognitionResponse>(url, { imagePath }, request);
    return {
      imageHeight: response.imageHeight,
      imageWidth: response.imageWidth,
      faces: response[ModelTask.FACIAL_RECOGNITION],
    };
  }

  async encodeImage(url: string, imagePath: string, { modelName }: CLIPConfig) {
    const request = { [ModelTask.SEARCH]: { [ModelType.VISUAL]: { modelName } } };
    const response = await this.predict<ClipVisualResponse>(url, { imagePath }, request);
    return response[ModelTask.SEARCH];
  }

  async encodeText(url: string, text: string, { modelName }: CLIPConfig) {
    const request = { [ModelTask.SEARCH]: { [ModelType.TEXTUAL]: { modelName } } };
    const response = await this.predict<ClipTextualResponse>(url, { text }, request);
    return response[ModelTask.SEARCH];
  }

  private async getFormData(payload: ModelPayload, config: MachineLearningRequest): Promise<FormData> {
    const formData = new FormData();
    formData.append('entries', JSON.stringify(config));

    if ('imagePath' in payload) {
      formData.append('image', new Blob([await readFile(payload.imagePath)]));
    } else if ('text' in payload) {
      formData.append('text', payload.text);
    } else {
      throw new Error('Invalid input');
    }

    return formData;
  }
}
