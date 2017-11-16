#ifndef _MATRIXHANDLER_
#define _MATRIXHANDLER_

#include "AdminHandler.h"
#include "Global.h"

class MatrixHandler: public AdminHandler {
public:
	MatrixHandler(string handler_factory);

	virtual void handle_create(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_update(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_read(World *world, string_t &path, http_request &request, http_response &response);
	virtual void handle_delete(World *world, string_t &path, http_request &request, http_response &response);

	//void create_up(World *world, http_request &request, http_response &response);
	void create_propagation(World *world, http_request &request, http_response &response);
	void create_npdirs(World *world, http_request &request, http_response &response);
	void create_up(World *world, http_request &request, http_response &response);
	void create_ups(World *world, http_request &request, http_response &response);
	void create_downs(World *world, http_request &request, http_response &response);
	void create_blocks(World *world, http_request &request, http_response &response);
	void create_abduction(World *world, http_request &request, http_response &response);
	void create_propagated_masks(World *world, http_request &request, http_response &response);
	~MatrixHandler();

protected:
	string_t UMA_BLOCK_DISTS, UMA_BLOCK_DT;
};

#endif
