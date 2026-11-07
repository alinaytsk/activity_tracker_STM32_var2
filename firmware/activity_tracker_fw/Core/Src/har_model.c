/* Реалізація інференсу. Див. har_model.h.
 *
 * Модуль навмисно тонкий: уся специфіка Cube.AI зібрана тут, а main.c бачить
 * лише har_model_predict(). Завдяки цьому заміна моделі (інша частота, інша
 * архітектура) не чіпає головний цикл.
 */
#include "har_model.h"

#include <stddef.h>

#if HAR_MODEL_DEPLOYED
/* Файли, згенеровані X-CUBE-AI. Якщо в CubeMX мережу названо не "network",
 * імена нижче зміняться відповідно (network -> <ім'я мережі>). */
#include "network.h"
#include "network_data.h"

static ai_handle  network = AI_HANDLE_NULL;
static ai_u8      activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
static ai_buffer *net_in;
static ai_buffer *net_out;
#endif

static const char *const class_names[HAR_CLASS_COUNT] = { HAR_CLASS_LIST };

const char *har_model_class_name(int cls)
{
	if (cls < 0 || cls >= HAR_CLASS_COUNT) {
		return "?";
	}
	return class_names[cls];
}

const char *har_model_status_str(har_model_status_t st)
{
	switch (st) {
	case HAR_MODEL_OK:           return "ok";
	case HAR_MODEL_NOT_DEPLOYED: return "not deployed";
	case HAR_MODEL_INIT_FAILED:  return "init failed";
	case HAR_MODEL_RUN_FAILED:   return "run failed";
	default:                     return "?";
	}
}

int har_model_available(void)
{
#if HAR_MODEL_DEPLOYED
	return network != AI_HANDLE_NULL;
#else
	return 0;
#endif
}

har_model_status_t har_model_init(void)
{
#if HAR_MODEL_DEPLOYED
	const ai_handle acts[] = { activations };

	ai_error err = ai_network_create_and_init(&network, acts, NULL);
	if (err.type != AI_ERROR_NONE) {
		network = AI_HANDLE_NULL;
		return HAR_MODEL_INIT_FAILED;
	}

	net_in  = ai_network_inputs_get(network, NULL);
	net_out = ai_network_outputs_get(network, NULL);
	if (net_in == NULL || net_out == NULL) {
		network = AI_HANDLE_NULL;
		return HAR_MODEL_INIT_FAILED;
	}
	return HAR_MODEL_OK;
#else
	return HAR_MODEL_NOT_DEPLOYED;
#endif
}

har_model_status_t har_model_predict(const float *window,
                                     float probs[HAR_CLASS_COUNT],
                                     int *top_class)
{
#if HAR_MODEL_DEPLOYED
	float local[HAR_CLASS_COUNT];
	float *out = (probs != NULL) ? probs : local;

	if (window == NULL) {
		return HAR_MODEL_RUN_FAILED;
	}
	if (network == AI_HANDLE_NULL) {
		return HAR_MODEL_INIT_FAILED;
	}

	/* Вхід уже у потрібному вигляді: HAR_WINDOW_SAMPLES x HAR_CHANNELS x 1
	 * float32, канали останні — рівно так навчалася модель, тож копіювати
	 * чи переставляти осі не треба. */
	net_in[0].data  = AI_HANDLE_PTR(window);
	net_out[0].data = AI_HANDLE_PTR(out);

	if (ai_network_run(network, net_in, net_out) != 1) {
		return HAR_MODEL_RUN_FAILED;
	}

	if (top_class != NULL) {
		int best = 0;
		for (int i = 1; i < HAR_CLASS_COUNT; i++) {
			if (out[i] > out[best]) {
				best = i;
			}
		}
		*top_class = best;
	}
	return HAR_MODEL_OK;
#else
	(void) window;

	/* Мережі немає: лишаємо вихід у визначеному стані, щоб той, хто
	 * забув перевірити код повернення, не читав сміття зі стека. */
	if (probs != NULL) {
		for (int i = 0; i < HAR_CLASS_COUNT; i++) {
			probs[i] = 0.0f;
		}
	}
	if (top_class != NULL) {
		*top_class = -1;
	}
	return HAR_MODEL_NOT_DEPLOYED;
#endif
}
